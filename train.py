# encoding:utf-8
import os
import time
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import BertModel, BertConfig, BertTokenizer
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model import SelectionDataset, SelectionSequentialTransform, SelectionJoinTransform, warmup_linear
from model import BertPolyDssmModel, BertDssmModel
from torch.nn import CrossEntropyLoss


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  # if args.n_gpu > 0:
  #   torch.cuda.manual_seed_all(args.seed)


def eval_running_model(dataloader):
  loss_fct = CrossEntropyLoss()
  model.eval()
  eval_loss, eval_hit_times = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0
  for step, batch in enumerate(dataloader, start=1):
    batch = tuple(t.to(device) for t in batch)
    context_token_ids_list_batch, context_segment_ids_list_batch, context_input_masks_list_batch, \
    response_token_ids_list_batch, response_segment_ids_list_batch, response_input_masks_list_batch, labels_batch = batch

    with torch.no_grad():
      logits = model(context_token_ids_list_batch, context_segment_ids_list_batch, context_input_masks_list_batch,
                     response_token_ids_list_batch, response_segment_ids_list_batch, response_input_masks_list_batch)
      loss = loss_fct(logits * 5, torch.argmax(labels_batch, 1))  # 5 is a coef

    eval_hit_times += (logits.argmax(-1) == torch.argmax(labels_batch, 1)).sum().item()
    eval_loss += loss.item()

    nb_eval_examples += labels_batch.size(0)
    nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  eval_accuracy = eval_hit_times / nb_eval_examples
  result = {
    'train_loss': tr_loss / nb_tr_steps,
    'eval_loss': eval_loss,
    'eval_accuracy': eval_accuracy,

    'epoch': epoch,
    'global_step': global_step,
  }
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  ## Required parameters
  # parser.add_argument("--bert_model", default='ckpt/pretrained/distilbert-base-uncased', type=str)
  # parser.add_argument("--model_type", default='distilbert', type=str)
  parser.add_argument("--bert_model", default='ckpt/pretrained/bert-small-uncased', type=str)
  parser.add_argument("--model_type", default='bert', type=str)
  parser.add_argument("--output_dir", required=True, type=str)
  parser.add_argument("--train_dir", default='data/ubuntu_data', type=str)

  parser.add_argument("--use_pretrain", action="store_true")
  parser.add_argument("--architecture", required=True, type=str, help='[poly, bi]')

  parser.add_argument("--max_contexts_length", default=128, type=int)
  parser.add_argument("--max_response_length", default=64, type=int)
  parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
  parser.add_argument("--eval_batch_size", default=2, type=int, help="Total batch size for eval.")
  parser.add_argument("--print_freq", default=100, type=int, help="Total batch size for eval.")

  parser.add_argument("--poly_m", default=16, type=int, help="Total batch size for eval.")
  parser.add_argument("--max_history", default=4, type=int, help="Total batch size for eval.")

  parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float)
  parser.add_argument("--warmup_steps", default=2000, type=float)
  parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

  parser.add_argument("--num_train_epochs", default=3.0, type=float,
                      help="Total number of training epochs to perform.")
  parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
  )
  parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
         "See details at https://nvidia.github.io/apex/amp.html",
  )
  parser.add_argument('--gpu', type=int, default=0)
  args = parser.parse_args()
  print(args)
  os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
  set_seed(args)

  MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer, BertModel),
    'distilbert': (DistilBertConfig, DistilBertTokenizer, DistilBertModel)
  }
  ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

  ## init dataset and bert model
  tokenizer = TokenizerClass.from_pretrained(os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True)
  context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=args.max_contexts_length,
                                             max_history=args.max_history)
  response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length,
                                                    max_history=None, pair_last=False)

  print('=' * 80)
  print('Train dir:', args.train_dir)
  print('Output dir:', args.output_dir)
  print('=' * 80)

  train_dataset = SelectionDataset(os.path.join(args.train_dir, 'train.txt'),
                                   context_transform, response_transform, sample_cnt=None)
  val_dataset = SelectionDataset(os.path.join(args.train_dir, 'test.txt'),
                                 context_transform, response_transform, sample_cnt=5000)
  train_dataloader = DataLoader(train_dataset,
                                batch_size=args.train_batch_size, collate_fn=train_dataset.batchify_join_str,
                                shuffle=True)
  val_dataloader = DataLoader(val_dataset,
                              batch_size=args.eval_batch_size, collate_fn=val_dataset.batchify_join_str, shuffle=False)
  t_total = len(train_dataloader) // args.train_batch_size * (max(5, args.num_train_epochs))

  epoch_start = 1
  global_step = 0
  best_eval_loss = float('inf')
  best_test_loss = float('inf')

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  shutil.copyfile(os.path.join(args.bert_model, 'vocab.txt'), os.path.join(args.output_dir, 'vocab.txt'))
  shutil.copyfile(os.path.join(args.bert_model, 'config.json'), os.path.join(args.output_dir, 'config.json'))
  log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8')

  state_save_path = os.path.join(args.output_dir, 'pytorch_model.bin')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  ########################################
  ## build BERT encoder
  ########################################
  bert_config = ConfigClass.from_json_file(os.path.join(args.bert_model, 'config.json'))
  if args.use_pretrain:
    previous_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
    print('Loading parameters from', previous_model_file)
    log_wf.write('Loading parameters from %s' % previous_model_file + '\n')
    model_state_dict = torch.load(previous_model_file, map_location="cpu")
    bert = BertModelClass.from_pretrained(args.bert_model, state_dict=model_state_dict)
    del model_state_dict
  else:
    bert = BertModelClass(bert_config)

  if args.architecture == 'poly':
    model = BertPolyDssmModel(bert_config, bert=bert, poly_m=args.poly_m)
  elif args.architecture == 'bi':
    model = BertDssmModel(bert_config, bert=bert)
  else:
    raise Exception('Unknown architecture.')
  model.to(device)

  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )
  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  tr_total = int(
    train_dataset.__len__() / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
  print_freq = args.print_freq
  eval_freq = min(len(train_dataloader) // 2, 1000)
  print('Print freq:', print_freq, "Eval freq:", eval_freq)

  for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    with tqdm(total=len(train_dataloader)) as bar:
      for step, batch in enumerate(train_dataloader, start=1):
        model.train()
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        context_token_ids_list_batch, context_segment_ids_list_batch, context_input_masks_list_batch, \
        response_token_ids_list_batch, response_segment_ids_list_batch, response_input_masks_list_batch, labels_batch = batch
        loss = model(context_token_ids_list_batch, context_segment_ids_list_batch, context_input_masks_list_batch,
                     response_token_ids_list_batch, response_segment_ids_list_batch, response_input_masks_list_batch,
                     labels_batch)
        tr_loss += loss.item()
        nb_tr_examples += context_token_ids_list_batch.size(0)
        nb_tr_steps += 1

        if args.fp16:
          with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        if global_step < args.warmup_steps:
          scheduler.step()
        model.zero_grad()
        optimizer.zero_grad()
        global_step += 1

        if step % print_freq == 0:
          bar.update(min(print_freq, step))
          time.sleep(0.02)
          print(global_step, tr_loss / nb_tr_steps)
          log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))

        if global_step % eval_freq == 0:
          if global_step == 4000:
            eval_freq *= 2
            print_freq *= 2
          if global_step == 16000:
            eval_freq *= 2
            print_freq *= 2

          scheduler.step()
          val_result = eval_running_model(val_dataloader)
          print('Global Step %d VAL res:\n' % global_step, val_result)
          log_wf.write('Global Step %d VAL res:\n' % global_step)
          log_wf.write(str(val_result) + '\n')

          if val_result['eval_loss'] < best_eval_loss:
            best_eval_loss = val_result['eval_loss']
            val_result['best_eval_loss'] = best_eval_loss
            # save model
            print('[Saving at]', state_save_path)
            log_wf.write('[Saving at] %s\n' % state_save_path)
            torch.save(model.state_dict(), state_save_path)
        log_wf.flush()
        pass

    # add a eval step after each epoch
    scheduler.step()
    val_result = eval_running_model(val_dataloader)
    print('Epoch %d, Global Step %d VAL res:\n' % (epoch, global_step), val_result)
    log_wf.write('Global Step %d VAL res:\n' % global_step)
    log_wf.write(str(val_result) + '\n')

    if val_result['eval_loss'] < best_eval_loss:
      best_eval_loss = val_result['eval_loss']
      val_result['best_eval_loss'] = best_eval_loss
      # save model
      print('[Saving at]', state_save_path)
      log_wf.write('[Saving at] %s\n' % state_save_path)
      torch.save(model.state_dict(), state_save_path)
    print(global_step, tr_loss / nb_tr_steps)
    log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))
