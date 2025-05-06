# Attention-Guide Masked for Implicit Relation Reasoning Aligning for Text-based Person Search

## Highlights
The goal of this work is to enhance global text-to-image person retrieval performance, without requiring any additional supervision and inference cost. To achieve this, we utilize the full CLIP model as our feature extraction backbone. Additionally, we propose a novel cross-modal matching loss (SDM) and an Implicit Relation Reasoning module to mine fine-grained image-text relationships, enabling IRRA to learn more discriminative global image-text representations.
![our AGMfIRRA Architecture](./AGMfIRRA/AGMfIRRA.png)
## Usage

### Requirements
We use a GPU T4 for training and evaluation.

- `pytorch 1.9.0`
- `torchvision 0.10.0`
- `prettytable`
- `easydict`

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), and the RSTPReid dataset from [here](https://github.com/NjtechCVLab/RSTPReid-Dataset).

## Training

```bash
python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'sdm+mlm+id+cmt' \
--dataset_name 'CUHK-PEDES' \
--root_dir 'your dataset root dir' \
--num_epoch 60
```
## Testing
```bash
python test.py --config_file 'path/to/model_dir/configs.yaml'
```
## Comparison with SOTA methods on CUHK-PEDES dataset
| Method              | Ref      | Image Enc. | Text Enc.    | Rank-1    | Rank-5    | Rank-10   | mAP       |
| ------------------- | -------- | ---------- | ------------ | --------- | --------- | --------- | --------- |
| CMPM/C              | ECCV18   | RN50       | LSTM         | 49.37     | -         | 79.27     | -         |
| TIMAM               | ICCV19   | RN101      | BERT         | 54.51     | 77.56     | 79.27     | -         |
| ViTAA               | ECCV20   | RN50       | LSTM         | 54.92     | 75.18     | 82.90     | 51.60     |
| NAFS                | arXiv21  | RN50       | BERT         | 59.36     | 79.13     | 86.00     | 54.07     |
| DSSL                | MM21     | RN50       | BERT         | 59.98     | 80.41     | 87.56     | -         |
| SSAN                | arXiv21  | RN50       | LSTM         | 61.37     | 80.15     | 86.73     | -         |
| LapsCore            | ICCV21   | RN50       | BERT         | 63.40     | -         | 87.80     | -         |
| ISANet              | arXiv22  | RN50       | LSTM         | 63.92     | 82.15     | 87.69     | -         |
| LBUL                | MM22     | RN50       | BERT         | 64.04     | 82.66     | 87.22     | -         |
| Han et al.          | BMVC21   | CLIP-RN101 | CLIP-Xformer | 64.08     | -         | 88.19     | 60.08     |
| SAF                 | ICASSP22 | ViT-Base   | BERT         | 64.13     | 82.62     | 88.40     | -         |
| TIPCB               | Neuro22  | RN50       | BERT         | 64.26     | 83.19     | 89.10     | -         |
| CAIBC               | MM22     | RN50       | BERT         | 64.43     | 83.12     | 88.67     | -         |
| AXM-Net             | MM22     | RN50       | BERT         | 64.44     | 82.80     | 86.77     | 58.73     |
| LGUR                | MM22     | DeiT-Small | BERT         | 65.25     | 83.12     | 88.71     | -         |
| IVT                 | ECCVW22  | ViT-Base   | BERT         | 65.59     | -         | 89.21     | -         |
| CFine               | arXiv22  | CLIP-ViT   | BERT         | 69.57     | 83.93     | 91.15     | -         |
| IRRA (Backbone)     | -        | CLIP-ViT   | CLIP-Xformer | 72.73     | 88.99     | 93.44     | 65.73     |
| **AGMfIRRA (Ours)** | -        | CLIP-ViT   | CLIP-Xformer | **73.46** | **89.41** | **93.44** | **65.73** |

### Acknowledgments
Some components of this code implementation are adopted from: [CLIP](https://github.com/openai/CLIP), [TextReID](https://github.com/BrandonHanx/TextReID), [TransReID](https://github.com/damo-cv/TransReID). We sincerely appreciate their contributions.

Contact
If you have any questions, please feel free to contact us via email: 21021292@vnu.edu.vn, 20021080@vnu.edu.vn, 22022542@vnu.edu.vn
