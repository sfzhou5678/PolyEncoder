# Poly-encoders

This repository is an unofficial implementation of [Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring](https://arxiv.org/abs/1905.01969v2).



## How to use

1. Download and unzip the ubuntu data https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntudata.zip?dl=0

2. Prepare a pretrained BERT (https://github.com/huggingface/transformers)

3. pip3 install -r requirements.txt 

4. Train a **Poly-encoder**:

   ```shell
   python3 train.py -bert_model /your/pretrained/model/dir --output_dir /your/ckpt/dir --train_dir /your/data/dir --use_pretrain --architecture poly --poly_m 16
   ```

4. Train a **Bi-encoder**:

   ```shell
   python3 train.py -bert_model /your/pretrained/model/dir --output_dir /your/ckpt/dir --train_dir /your/data/dir --use_pretrain --architecture bi
   ```

   

## Results

The experimental settings and results can be shown as follows:

- **Dataset**: Ubuntu V2
- **Device**: GTX 1060 6G x1
- **Pretrained model:** BERT-small-uncased (https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip )
- **Batch size:** 32
- **max_contexts_length**: 128
- **max_context_cnt:** 4
- **max_response_length**：64
- **lr**: 5e-5
- **Epochs**: 3

|       Model       | **R@1/10** | **Training Speed** | **GPU Mem Consumption** |
| :---------------: | :--------: | :----------------: | :---------------------: |
|    Bi-encoder     |   0.6714   |      3.15it/s      |        1969  Mb         |
| Poly-encoder  16  |   0.6938   |      3.11it/s      |         1975Mb          |
| Poly-encoder  64  |            |      3.08it/s      |         2005Mb          |
| Poly-encoder  360 |            |                    |                         |



Different with the original paper, this experiment uses a **bert-small-uncased** model (from https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip) rather than the **bert-base**. Beside, this experiment only uses **batch_size =32, max_length = 128, and max_history=4** (which means select up to 4 context texts). All these settings lead to lower results, but faster training speeds. One can modify these settings for a better results.

