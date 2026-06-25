# DEER: A Foundation Model with Dual-Branch CT-Enhanced Embeddings for Thoracoabdominal Radiographs

This project develops a medical foundation model for digital radiography (DR) through large-scale multi-modal contrastive learning. We propose a unified dual-branch pretraining framework that jointly leverages 1.1M chest and abdominal DR images and 1M CT localizers (LOC), each paired with radiologist reports.

LOC, despite being widely available in clinical workflows, have been largely underexplored in prior research. They are visually and structurally similar to DR images, while being naturally aligned with CT reports, making them a unique and effective bridge for transferring CT-level semantic knowledge into radiographic representation learning. By incorporating LOC-report pairs into a contrastive learning framework, the model produces more discriminative and clinically meaningful DR embeddings.

Extensive evaluations across 10 downstream tasks demonstrate the superiority of DEER and highlight the effectiveness of CT-enhanced contrastive learning for robust DR foundation modeling.

## ✨ Highlights

- [🧠 Training](#-training)
- [🚀 Deployment](#-deployment)
- [🌐 Online API](#-online-api)

## 📊 Preparing Datasets

The dataset is provided as a tab-separated CSV file (`.csv`).

### Data Format
Each row in the dataset corresponds to one sample and contains the following fields:

- `filepath`: Path to the image file.
- `title`: The corresponding radiology report describing the image.
- `modality`: Integer label indicating imaging modality:
  - `0`: DR
  - `1`: LOC

### Dataset Location
The dataset file is stored at:
```
/path/to/data.csv
```

### File Format Details
- Separator: `\t` (tab-separated values)
- Encoding: UTF-8 (recommended)

### Example
```
filepath  title  modality
/path/to/img/0001.png  No acute cardiopulmonary abnormality.  0
/path/to/img/0002.png  Mild opacity observed in the lower lung.  1
```

### Checking the Dataset

You can load and check the dataset using `pandas`:

```python
import pandas as pd

data_path = "/path/to/data.csv"
df = pd.read_csv(data_path, sep="\t")

print(df.head())
```

## 🧠 Training
This project is built on top of OpenCLIP and PyTorch. We recommend using a clean virtual environment (e.g., conda or venv).

### Core Training Dependencies
- Python=3.12
- PyTorch
- torchvision
- pandas
- numpy
- pillow

### Installation
You can install all required training dependencies and the local project as follows:

```bash
conda create -n my_env python=3.12
git clone https://github.com/sunyh1/DEER.git
cd DEER
pip install torch==2.8.0 torchvision==0.23.0 tensorboard
make install
make install-training
```

### Single-GPU Training
```bash
cd src
python -m open_clip_train.main \
--train-data '/path/to/data.csv' \
--dataset-type csv \
--batch-size 512 \
--precision amp_bf16 \
--workers 8 \
--model ViT-L-14-336 \
--pretrained openai \
--force-image-size 518 \
--force-context-length 128 \
--lr 3e-5 \
--epochs 3 \
--warmup 200 \
--grad-checkpointing \
--accum-freq 2 \
--report-to tensorboard \
--image-resize-mode squash
```

### Multi-GPU Training
```bash
cd src
torchrun --nproc_per_node 2 -m open_clip_train.main \
--train-data '/path/to/data.csv' \
--dataset-type csv \
--batch-size 512 \
--precision amp_bf16 \
--workers 8 \
--model ViT-L-14-336 \
--pretrained openai \
--force-image-size 518 \
--force-context-length 128 \
--lr 3e-5 \
--epochs 3 \
--warmup 200 \
--grad-checkpointing \
--accum-freq 2 \
--report-to tensorboard \
--image-resize-mode squash
```

## 🚀 Deployment

For deployment, we recommend creating a new clean environment to avoid conflicts with training dependencies. 

Then install OpenCLIP (inference only):

```bash
pip install open_clip_torch
```

### Convert Model Weights
First, convert the trained DR branch weights into an OpenCLIP-compatible format using our provided script:

```bash
python convert_weights.py
```

This step ensures the model weights can be correctly loaded by OpenCLIP.

### Feature Extraction Example

The following example shows how to extract image and text features and compute similarity:

```
import torch
from PIL import Image
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'ViT-L-14-336',
    pretrained='/path/to/checkpoint_dr.pt',
    force_image_size=518,
    force_context_length=128
)

model = model.to(device)
model.eval()

# --------------------
# Image feature extraction
# --------------------
image = preprocess(Image.open("example.jpg")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# --------------------
# Text feature extraction
# --------------------
text = tokenizer(["normal examination with on abnormality."]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# --------------------
# Similarity computation
# --------------------
similarity = (image_features @ text_features.T)
print(similarity)
```

## 🌐 Online API
The model is now available through [MedHere AI](https://sjtu.pacsonline.cn), our platform for medical foundation models. The platform interface is available in both English and Chinese.

You are welcome to register an account and submit an application for API access. Once approved, you will be able to integrate and use the model through our API services.

Currently, the model is available through our API. We do not have permission to release the model weights publicly at this time. Additional medical foundation models will be deployed and made available through [MedHere AI](https://sjtu.pacsonline.cn) in the future. Please stay tuned for updates. ✨

## 🙏 Acknowledgements

This project is built on top of [OpenCLIP](https://github.com/mlfoundations/open_clip) and [PyTorch](https://pytorch.org/).

## 📚 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{sun2026deer,
  title={DEER: A Foundation Model with Dual-Branch CT-Enhanced Embeddings for Thoracoabdominal Radiographs},
  author={Yihua Sun, Jia Guo, Qijing Wang, Jingzhou Ouyang, Ge Li, Nan Zhang, Fang Chen, and Hongen Liao},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2026}
}
