import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, DistilBertModel
from transformers.modeling_bert import BertEncoder, BertPooler, BertLayerNorm
from model import dot_attention


class BertDssmModel(BertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.bert = kwargs['bert']
    try:
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.context_fc = nn.Linear(config.hidden_size, 64)
      self.response_fc = nn.Linear(config.hidden_size, 64)
    except:
      self.dropout = nn.Dropout(config.dropout)
      self.context_fc = nn.Linear(config.dim, 64)
      self.response_fc = nn.Linear(config.dim, 64)

  def forward(self, context_input_ids, context_segment_ids, context_input_masks,
              responses_input_ids, responses_segment_ids, responses_input_masks, labels=None):
    ## only select the first response (whose lbl==1)
    if labels is not None:
      responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
      responses_segment_ids = responses_segment_ids[:, 0, :].unsqueeze(1)
      responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)

    if isinstance(self.bert, DistilBertModel):
      context_vec = self.bert(context_input_ids, context_input_masks)[-1]  # [bs,dim]
      context_vec = context_vec[:, 0]
    else:
      context_vec = self.bert(context_input_ids, context_input_masks, context_segment_ids)[-1]  # [bs,dim]

    batch_size, res_cnt, seq_length = responses_input_ids.shape
    responses_input_ids = responses_input_ids.view(-1, seq_length)
    responses_input_masks = responses_input_masks.view(-1, seq_length)
    responses_segment_ids = responses_segment_ids.view(-1, seq_length)

    if isinstance(self.bert, DistilBertModel):
      responses_vec = self.bert(responses_input_ids, responses_input_masks)[-1]  # [bs,dim]
      responses_vec = responses_vec[:, 0]
    else:
      responses_vec = self.bert(responses_input_ids, responses_input_masks, responses_segment_ids)[
        -1]  # [bs,dim]
    responses_vec = responses_vec.view(batch_size, res_cnt, -1)

    context_vec = self.context_fc(self.dropout(context_vec))
    context_vec = F.normalize(context_vec, 2, -1)

    responses_vec = self.response_fc(self.dropout(responses_vec))
    responses_vec = F.normalize(responses_vec, 2, -1)

    if labels is not None:
      responses_vec = responses_vec.squeeze(1)
      dot_product = torch.matmul(context_vec, responses_vec.t())  # [bs, bs]
      mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
      loss = F.log_softmax(dot_product * 5, dim=-1) * mask
      loss = (-loss.sum(dim=1)).mean()

      return loss
    else:
      context_vec = context_vec.unsqueeze(1)
      dot_product = torch.matmul(context_vec, responses_vec.permute(0, 2, 1))  # take this as logits
      dot_product.squeeze_(1)
      cos_similarity = (dot_product + 1) / 2
      return cos_similarity


class BertPolyDssmModel(BertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.bert = kwargs['bert']

    self.vec_dim = 64

    self.poly_m = kwargs['poly_m']
    self.poly_code_embeddings = nn.Embedding(self.poly_m + 1, config.hidden_size)
    self.poly_context_fc = nn.Linear(self.vec_dim, self.vec_dim)
    self.poly_dropout = nn.Dropout(0.5)
    try:
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.context_fc = nn.Linear(config.hidden_size, self.vec_dim)
      self.response_fc = nn.Linear(config.hidden_size, self.vec_dim)
    except:
      self.dropout = nn.Dropout(config.dropout)
      self.context_fc = nn.Linear(config.dim, self.vec_dim)
      self.response_fc = nn.Linear(config.dim, self.vec_dim)

  def forward(self, context_input_ids, context_segment_ids, context_input_masks,
              responses_input_ids, responses_segment_ids, responses_input_masks, labels=None):
    ## only select the first response (whose lbl==1)
    if labels is not None:
      responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
      responses_segment_ids = responses_segment_ids[:, 0, :].unsqueeze(1)
      responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
    batch_size, res_cnt, seq_length = responses_input_ids.shape

    ## poly context encoder
    if isinstance(self.bert, DistilBertModel):
      state_vecs = self.bert(context_input_ids, context_input_masks)[-1]  # [bs, length, dim]
    else:
      state_vecs = self.bert(context_input_ids, context_input_masks, context_segment_ids)[0]  # [bs, length, dim]
    poly_code_ids = torch.arange(self.poly_m, dtype=torch.long, device=context_input_ids.device)
    poly_code_ids += 1
    poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
    poly_codes = self.poly_code_embeddings(poly_code_ids)
    # attention_weights = torch.matmul(poly_codes, state_vecs.transpose(-1, -2))
    # attention_weights *= context_input_masks.unsqueeze(1)
    # attention_weights = F.softmax(attention_weights, -1)
    # attention_weights = self.dropout(attention_weights)
    # context_vecs = torch.matmul(attention_weights, state_vecs)
    context_vecs = dot_attention(poly_codes, state_vecs, state_vecs, context_input_masks, self.dropout)

    ## response encoder
    responses_input_ids = responses_input_ids.view(-1, seq_length)
    responses_input_masks = responses_input_masks.view(-1, seq_length)
    responses_segment_ids = responses_segment_ids.view(-1, seq_length)
    if isinstance(self.bert, DistilBertModel):
      state_vecs = self.bert(responses_input_ids, responses_input_masks)[-1]  # [bs, length, dim]
    else:
      state_vecs = self.bert(responses_input_ids, responses_input_masks, responses_segment_ids)[0]  # [bs, length, dim]
    poly_code_ids = torch.zeros(batch_size * res_cnt, 1, dtype=torch.long, device=context_input_ids.device)
    poly_codes = self.poly_code_embeddings(poly_code_ids)
    # attention_weights = torch.matmul(poly_codes, state_vecs.transpose(-1, -2))
    # attention_weights *= responses_input_masks.unsqueeze(1)
    # attention_weights = F.softmax(attention_weights, -1)
    # attention_weights = self.dropout(attention_weights)
    # responses_vec = torch.matmul(attention_weights, state_vecs)
    responses_vec = dot_attention(poly_codes, state_vecs, state_vecs, responses_input_masks, self.dropout)
    responses_vec = responses_vec.view(batch_size, res_cnt, -1)

    ## 这里先norm一下，相当于以某种方式得到了context_vec和response_vec
    context_vecs = self.context_fc(self.dropout(context_vecs))
    context_vecs = F.normalize(context_vecs, 2, -1)  # [bs, m, dim]
    responses_vec = self.response_fc(self.dropout(responses_vec))
    responses_vec = F.normalize(responses_vec, 2, -1)

    ## poly final context vector aggregation
    if labels is not None:
      responses_vec = responses_vec.view(1, batch_size, -1).expand(batch_size, batch_size, self.vec_dim)
    # attention_weights = F.softmax(torch.matmul(responses_vec, context_vecs.transpose(-1, -2)), -1)
    # attention_weights = self.dropout(attention_weights)
    # final_context_vec = torch.matmul(attention_weights, context_vecs)
    final_context_vec = dot_attention(responses_vec, context_vecs, context_vecs, None, self.dropout)
    final_context_vec = F.normalize(final_context_vec, 2, -1)  # [bs, res_cnt, dim], res_cnt==bs when training

    dot_product = torch.sum(final_context_vec * responses_vec, -1)  # [bs, res_cnt], res_cnt==bs when training
    if labels is not None:
      mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
      loss = F.log_softmax(dot_product * 5, dim=-1) * mask
      loss = (-loss.sum(dim=1)).mean()

      return loss
    else:
      cos_similarity = (dot_product + 1) / 2
      return cos_similarity
