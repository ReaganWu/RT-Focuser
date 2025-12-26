# RT-Focuser: Real-Time Lightweight Model for Edge-side Image Deblurring

[![Paper](https://img.shields.io/badge/Paper-ICTA%202025-blue)](https://github.com/ReaganWu/RT-Focuser)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-red)](https://arxiv.org/abs/xxxx.xxxxx)
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
![image](https://github.com/ReaganWu/RT-Focuser/blob/main/IMG/comparison_vertical.gif)

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

![image](https://github.com/ReaganWu/RT-Focuser/blob/main/IMG/RT-Focuser_Arch.png)
---

## 📊 Performance

### Quantitative Results on GoPro Dataset

| Model | PSNR↑ | SSIM↑ | Params↓ | GMACs↓ | Time (s)↓ |
|-------|-------|-------|---------|--------|-----------|
| SRN | 29.97 | 0.9013 | 8.06M | 109.07 | 2.52 |
| MIMO-UNet | 31.73 | 0.9500 | 16.10M | 154.41 | 0.014 |
| DeepDeblur | 29.23 | 0.9160 | 11.70M | 62.85 | 4.33 |
| EDVR | 31.54 | 0.9260 | 23.61M | 33.44 | 0.21 |
| STSM | 33.41 | 0.9512 | 14.40M | 92.51 | 0.16 |
| **RT-Focuser** | **30.67** | **0.9005** | **5.85M** | **15.76** | **0.006** |

✨ **RT-Focuser achieves 100× speedup compared to large models while maintaining competitive quality!**

### Multi-Platform Deployment Speed

| Platform | FPS↑ | Backend | Details |
|----------|------|---------|---------|
| **iPhone 15 (A16 Bionic)** | **146.72** | CoreML | Mobile deployment |
| **RTX 3090 GPU** | **154.42** | PyTorch CUDA | Desktop inference |
| **Intel Xeon CPU** | **22.74** | OpenVINO | CPU optimization |
| Intel Xeon CPU | 14.95 | ONNX Runtime | General backend |

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
wget https://github.com/ReaganWu/RT-Focuser/releases/download/v1.0/rt_focuser_gopro.pth \
  -O checkpoints/rt_focuser_gopro.pth
```

### Inference on Single Image

```bash
python inference.py \
  --input_path ./test_images/blurry.png \
  --output_path ./results/sharp.png \
  --checkpoint ./checkpoints/rt_focuser_gopro.pth \
  --device cuda
```

### Python API Usage

```python
import torch
from models.rt_focuser import RTFocuser
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = RTFocuser()
checkpoint = torch.load('checkpoints/rt_focuser_gopro.pth')
model.load_state_dict(checkpoint['model'])
model.eval().cuda()

# Prepare image
transform = transforms.ToTensor()
blurry_img = Image.open('blurry.png')
input_tensor = transform(blurry_img).unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    sharp_img = model(input_tensor)
```

---

## 📦 Model Export

### Export to ONNX

```bash
python export_onnx.py \
  --checkpoint ./checkpoints/rt_focuser_gopro.pth \
  --output ./exports/rt_focuser.onnx \
  --input_size 256 256
```

### Export to CoreML (iOS/macOS)

```bash
python export_coreml.py \
  --checkpoint ./checkpoints/rt_focuser_gopro.pth \
  --output ./exports/RTFocuser.mlmodel
```

### Quantization (INT8)

```bash
python quantize_int8.py \
  --checkpoint ./checkpoints/rt_focuser_gopro.pth \
  --calibration_data ./data/GoPro/train/ \
  --output ./exports/rt_focuser_int8.onnx
```

---

## 🎓 Training

### Prepare GoPro Dataset

```bash
# Download dataset
wget http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large.zip
unzip GOPRO_Large.zip -d ./data/
```

### Train from Scratch

```bash
python train.py \
  --data_root ./data/GoPro \
  --batch_size 16 \
  --patch_size 256 \
  --epochs 3000 \
  --lr 1e-4
```

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
@inproceedings{wu2025rtfocuser,
  title={RT-Focuser: A Real-Time Lightweight Model for Edge-side Image Deblurring},
  author={Wu, Zhuoyu and Ou, Wenhui and Zheng, Qiawei and Yang, Jiayan and Wang, Quanjun and Fang, Wenqi and Wang, Zheng and Yang, Yongkui and Li, Heshan},
  booktitle={International Conference on Integrated Circuits, Technologies and Applications (ICTA)},
  year={2025}
}
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