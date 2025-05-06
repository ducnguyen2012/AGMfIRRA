# Attention-Guide Masked for Implicit Relation Reasoning Aligning for Text-based Person Search

## Highlights
The goal of this work is to enhance global text-to-image person retrieval performance, without requiring any additional supervision and inference cost. To achieve this, we utilize the full CLIP model as our feature extraction backbone. Additionally, we propose a novel cross-modal matching loss (SDM) and an Implicit Relation Reasoning module to mine fine-grained image-text relationships, enabling IRRA to learn more discriminative global image-text representations.

## Usage
### Requirements
we use single RTX3090 24G GPU for training and evaluation.

pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
## Prepare Datasets
Download the CUHK-PEDES dataset from [here]([url](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)), RSTPReid dataset form [here]([url](https://github.com/NjtechCVLab/RSTPReid-Dataset))

Organize them in your dataset root dir folder as follows:

|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
##Training
python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'sdm+mlm+id' \
--dataset_name 'CUHK-PEDES' \
--root_dir 'your dataset root dir' \
--num_epoch 60

## Testing
python test.py --config_file 'path/to/model_dir/configs.yaml'
IRRA on Text-to-Image Person Retrieval Results
Compare with SOTA method in CUHK-PEDES dataset
Method Ref Image Enc. Text Enc. Rank-1 Rank-5 Rank-10 mAP
CMPM/C ECCV18 RN50 LSTM 49.37 - 79.27 -
TIMAM ICCV19 RN101 BERT 54.51 77.56 79.27 -
ViTAA ECCV20 RN50 LSTM 54.92 75.18 82.90 51.60
NAFS arXiv21 RN50 BERT 59.36 79.13 86.00 54.07
DSSL MM21 RN50 BERT 59.98 80.41 87.56 -
SSAN arXiv21 RN50 LSTM 61.37 80.15 86.73 -
LapsCore ICCV21 RN50 BERT 63.40 - 87.80 -
ISANet arXiv22 RN50 LSTM 63.92 82.15 87.69 -
LBUL MM22 RN50 BERT 64.04 82.66 87.22 -
Han et al. BMVC21 CLIP-RN101 CLIP-Xformer 64.08 - 88.19 60.08
SAF ICASSP22 ViT-Base BERT 64.13 82.62 88.40 -
TIPCB Neuro22 RN50 BERT 64.26 83.19 89.10 -
CAIBC MM22 RN50 BERT 64.43 83.12 88.67 -
AXM-Net MM22 RN50 BERT 64.44 82.80 86.77 58.73
LGUR MM22 DeiT-Small BERT 65.25 83.12 88.71 -
IVT ECCVW22 ViT-Base BERT 65.59 - 89.21 -
CFine arXiv22 CLIP-ViT BERT 69.57 83.93 91.15 -
IRRA (Backbone) - CLIP-ViT CLIP-Xformer 72.726 88.986 93.437 65.729
AGMfIRRA (our) - CLIP-ViT CLIP-Xformer 73.457 89.409 93.437 65.729


Acknowledgments
Some components of this code implementation are adopted from [CLIP]([url](https://github.com/openai/CLIP)), [TextReID]([url](https://github.com/BrandonHanx/TextReID)) and [TransReID]([url](https://github.com/damo-cv/TransReID)). We sincerely appreciate for their contributions.

Contact
If you have any question, please feel free to contact us. E-mail: 21021292@vnu.edu.vn, 20021080@vnu.edu.vn, 22022542@vnu.edu.vn .
