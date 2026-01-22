# RT-Focuser: Real-Time Lightweight Model for Edge-side Image Deblurring

[![Paper](https://img.shields.io/badge/Paper-ICTA%202025-blue)](https://ieeexplore.ieee.org/document/11329854)
[![Arxiv](https://img.shields.io/badge/arXiv-2512.21975-red)](https://arxiv.org/abs/2512.21975)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official PyTorch implementation of RT-Focuser (ICTA 2025)**

> **RT-Focuser: A Real-Time Lightweight Model for Edge-side Image Deblurring**
> Zhuoyu Wu, Wenhui Ou, Qiawei Zheng, Jiayan Yang, Quanjun Wang, Wenqi Fang, Zheng Wang, Yongkui Yang, Heshan Li
> *Accepted to IEEE ICTA 2025*

---

![image](https://github.com/ReaganWu/RT-Focuser/blob/main/IMG/RT-Focuser_Perf.png)

---

## 🔥 Highlights

- **⚡ Ultra-Fast**: 6ms per frame on RTX 3090, **146 FPS on iPhone 15**
- **🪶 Lightweight**: Only **5.85M parameters** and **15.76 GMACs**
- **📱 Edge-Ready**: Deployable on mobile devices with CoreML
- **🎯 Single-Input-Single-Output**: Efficient for real-time video streaming

---

![image](https://github.com/ReaganWu/RT-Focuser/blob/main/IMG/comparison_horizontal.gif)

## 🎯 Introduction

Motion blur from camera or object movement severely degrades image quality and poses challenges for real-time applications such as:

- 🚗 Autonomous driving
- 🚁 UAV perception
- 🏥 Medical imaging
- 📹 Video streaming

**RT-Focuser** is designed specifically for **real-time edge deployment** with a balanced trade-off between speed and quality.

### Key Features

1. **Lightweight Deblurring Block (LD)**: Edge-aware feature extraction with sharpness normalization
2. **Multi-Level Integrated Aggregation (MLIA)**: Hierarchical encoder feature fusion
3. **Cross-source Fusion Block (X-Fuse)**: Progressive decoder refinement with multi-source inputs

## 📊 Performance

### Quantitative Results on GoPro Dataset

| Model                | PSNR↑          | SSIM↑           | Params↓        | GMACs↓         | Time (s)↓      |
| -------------------- | --------------- | ---------------- | --------------- | --------------- | --------------- |
| SRN                  | 29.97           | 0.9013           | 8.06M           | 109.07          | 2.52            |
| MIMO-UNet            | 31.73           | 0.9500           | 16.10M          | 154.41          | 0.014           |
| DeepDeblur           | 29.23           | 0.9160           | 11.70M          | 62.85           | 4.33            |
| EDVR                 | 31.54           | 0.9260           | 23.61M          | 33.44           | 0.21            |
| STSM                 | 33.41           | 0.9512           | 14.40M          | 92.51           | 0.16            |
| **RT-Focuser** | **30.67** | **0.9005** | **5.85M** | **15.76** | **0.006** |

✨ **RT-Focuser achieves 100× speedup compared to large models while maintaining competitive quality!**

### Multi-Platform Deployment Speed

| Platform                         | FPS↑            | Backend      | Details           |
| -------------------------------- | ---------------- | ------------ | ----------------- |
| **iPhone 15 (A16 Bionic)** | **146.72** | CoreML       | Mobile deployment |
| **RTX 3090 GPU**           | **154.42** | PyTorch CUDA | Desktop inference |
| **Intel Xeon CPU**         | **22.74**  | OpenVINO     | CPU optimization  |
| Intel Xeon CPU                   | 14.95            | ONNX Runtime | General backend   |

*Note: Measured on 256×256 input, batch size 1*

---

## 🚀 Installation

### Requirements

```bash
Python >= 3.8
PyTorch >= 1.12.0
CUDA >= 11.3 (for GPU)
```

### Clone Repository

```bash
git clone https://github.com/ReaganWu/RT-Focuser.git
cd RT-Focuser
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎮 Quick Start

### Download Pretrained Weights

```bash
# Download from release page
wget https://github.com/ReaganWu/RT-Focuser/blob/main/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth \
  -O rt_focuser_gopro.pth
```

### Test RT-Focuser on Single Image

Pytorch Version

> Input: *Sample/Blurry.png*
>
> Output: *Sample/Deblur_F32.png*

```bash
python Inference_Image_Torch.py
```

ONNX Version

> Input: *Sample/Blurry.png*
>
> Output: *Sample/Deblur_W8A16.png*

```bash
python Inference_Image_ONNX.py
```

### RT-Focuser CLI Usage via ONNX (Image, Video)

Image Inference

```bash
python onnx_image_inference.py \
  --model Pretrained_Weights/rt_focuser_wint8_afp16.onnx \
  --input Sample/Blurry.png \
  --output Sample/Deblur_W8A16.png \
  --width 256 \
  --height 256
```

Video Inference

```bash
python onnx_video_inference.py \
  --model Pretrained_Weights/rt_focuser_wint8_afp16.onnx \
  --input Test.mp4 \
  --output output.mp4
```

### Python API Usage

```python
import torch
from model.rt_focuser import RTFocuser
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = RTFocuser()
checkpoint = torch.load('Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth')
model.load_state_dict(checkpoint['model'])
model.eval().cuda()

# Prepare image
transform = transforms.ToTensor()
blurry_img = Image.open('Sample/Blurry.png')
input_tensor = transform(blurry_img).unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    sharp_img = model(input_tensor)
```

---

## 🛸 Training

A minimal and clean training framework for image deblurring on GoPro dataset.

### Requirements

```bash
pip install torch torchvision
pip install albumentations
pip install scikit-image
pip install Pillow
pip install tqdm
```

### Dataset Structure

Your GoPro dataset should be organized as:

```
GOPRO_Large/
├── train/
│   ├── GOPR0372_07_00/
│   │   ├── blur_gamma/
│   │   └── sharp/
│   ├── GOPR0372_07_01/
│   └── ...
└── test/
    ├── GOPR0384_11_00/
    └── ...
```

### Quick Start

1. **Prepare your model**: Implement your deblurring model

2. **Configure paths**: Edit `train.py` to set your dataset path

3. **Run training**:
```bash
python train.py
```

4. **Checkpoints**: Models will be saved to `./Pretrained_Weights/`
   - `best_model.pth`: Best model based on PSNR
   - `checkpoint_epoch_X.pth`: Checkpoint every 100 epochs

### Training Configuration

Default settings in `train.py`:
- Learning rate: **1e-4**
- Epochs: **3000**
- Batch size: 4
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Crop size: 256×256
- Data augmentation: Random crop + Horizontal flip

### File Structure

- `setup\dataset.py`: GoPro dataset loader
- `setup\trainer.py`: Training and validation functions
- `train.py`: Main training script (modify this for your use case)

### Example Training Code

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataloaders
from trainer import train_model

# Your model
from rt_focuser import RTFocuser

# Configuration
DATA_ROOT = "/path/to/your/gopro/dataset"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
train_loader, val_loader = get_dataloaders(
    root_dir=DATA_ROOT,
    batch_size=4,
    num_workers=4
)

# Initialize model
model = RTFocuser()

# Setup optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-6)

# Train
best_psnr = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    num_epochs=3000,
    device=DEVICE,
    save_dir='./Pretrained_Weights'
)
```

---

## ⚙️ Hybrid Quantization (Configurable)

RT-Focuser supports configurable hybrid post-training quantization:

* Weight precision: `fp32 | int8`
* Activation precision: `fp32 | fp16 | int8`
* Compatible with ONNX export & deployment

The interface is unified — just set the quantization modes you want.

❕should be noticed that, you should download GoPro or your self-prepared dataset before using the quantization program.

```python
from torch.utils.data import DataLoader
from rt_focuser import RTFocuser
from PTQ_Transfer_ONNX import (
    GoProFullSizeDataset,
    CalibrationDataset,
    apply_hybrid_quantization_v2
)

import torch

# 1) Load model checkpoint
model = RTFocuser()
ckpt = torch.load("./checkpoints/rt_focuser_gopro.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

# 2) Build calibration dataloader
fullset = GoProFullSizeDataset("./data/GoPro", mode="test")
calibset = CalibrationDataset(fullset, num_samples=100, patch_size=256)
calib_loader = DataLoader(calibset, batch_size=4, shuffle=True)

# 3) Choose quantization configuration
model = apply_hybrid_quantization_v2(
    model=model,
    calibration_loader=calib_loader,
    device="cuda",

    # Supported modes:
    #   weight_dtype:  fp32 | int8
    #   activation_dtype: fp32 | fp16 | int8
    weight_dtype="int8",
    activation_dtype="fp16"
)

# 4) Export quantized ONNX model
dummy = torch.randn(1, 3, 256, 256).cuda().half()
torch.onnx.export(
    model,
    dummy,
    "./exports/rt_focuser_hybrid.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"]
)
```

---

### ✔️ Available Quantization Modes

| Weights  | Activations | Usage Scenario                                  |
| -------- | ----------- | ----------------------------------------------- |
| `fp32` | `fp32`    | baseline / reference                            |
| `int8` | `fp32`    | low-risk compression                            |
| `int8` | `fp16`    | **hybrid performance mode (recommended)** |
| `int8` | `int8`    | aggressive compression                          |
| `fp32` | `fp16`    | GPU / CoreML friendly                           |

Switch modes by changing parameters:

```python
weight_dtype="int8"
activation_dtype="fp16"
```

No other code changes are needed.

---

### 📤 Output

The exported ONNX model contains the quantized weights and activation precision configuration and can be used for:

* TensorRT
* ONNX Runtime
* RKNN
* Edge devices
* Mobile / CoreML (FP16 friendly)

---

## 🔬 Extended Version (Coming Soon)

We are preparing an **extended arXiv version** with:

- ✅ Comprehensive quantization analysis (PTQ/QAT)
- ✅ Out-of-distribution generalization
- ✅ Multi-platform deployment details
- ✅ Energy efficiency analysis

---

## 📝 Citation

```bibtex
@misc{wu2025rtfocuserrealtimelightweightmodel,
      title={RT-Focuser: A Real-Time Lightweight Model for Edge-side Image Deblurring}, 
      author={Zhuoyu Wu and Wenhui Ou and Qiawei Zheng and Jiayan Yang and Quanjun Wang and Wenqi Fang and Zheng Wang and Yongkui Yang and Heshan Li},
      year={2025},
      eprint={2512.21975},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2512.21975}, 
}

@INPROCEEDINGS{IEEE_wu2025rtfocuser,
  author={Wu, Zhuoyu and Ou, Wenhui and Zheng, Qiawei and Yang, Jiayan and Wang, Quanjun and Fang, Wenqi and Wang, Zheng and Yang, Yongkui and Li, Heshan},
  booktitle={2025 IEEE International Conference on Integrated Circuits, Technologies and Applications (ICTA)}, 
  title={RT-Focuser: A Real-Time Lightweight Model for Edge-Side Image Deblurring}, 
  year={2025},
  volume={},
  number={},
  pages={255-256},
  keywords={Deblurring;Integrated circuit technology;Image quality;Image edge detection;Computational modeling;Graphics processing units;Feature extraction;Real-time systems;Decoding;Integrated circuit modeling;Image Deblurring;Real-Time Inference;Lightweight Network;Edge Deployment},
  doi={10.1109/ICTA68203.2025.11329854}}
```

---

## 📧 Contact

- **Email**: wuzhuoyu11@gmail.com
- **Issues**: Please open an issue in this repository

---

## 🙏 Acknowledgments

This work was funded by National Science Foundation of China (NSFC) under Grant No.12401676 and No.62372442.

---

## 📄 License

This project is released under the MIT License.
