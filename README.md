# SRGAN DIV2K - Image Super-Resolution with Generative Adversarial Networks

A comprehensive implementation of **Super-Resolution Generative Adversarial Networks (SRGAN)** trained on the **DIV2K dataset** for high-quality image upsampling and restoration.

## 📋 Overview

This project implements SRGAN, a deep learning approach for image super-resolution that uses adversarial training to generate high-quality high-resolution images from low-resolution inputs. The model is trained on the DIV2K (Diverse 2K resolution) dataset, a popular benchmark for image super-resolution tasks.

### Key Features
- **SRGAN Architecture**: Generator and Discriminator networks for adversarial training
- **Perceptual Loss**: Uses VGG features to preserve image content and style
- **Residual Blocks**: Deep residual network for efficient feature extraction
- **DIV2K Dataset**: Trained on diverse, high-quality 2K resolution images
- **Performance Metrics**: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
- **Training Visualization**: Jupyter notebooks for monitoring training curves and metrics

## 🗂️ Project Structure

```
srgan-div2k/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 02_training_curves.ipynb      # Visualize training metrics
│   └── 03_benchmark_metrics.ipynb    # Evaluate model performance
├── src/                      # Source code
│   ├── train/               # Training scripts and model definitions
│   │   └── model.py         # Generator and Discriminator architectures
│   ├── data/                # Dataset handling and preprocessing
│   └── utils/               # Utility functions
├── outputs/                 # Generated outputs and logs
│   └── logs/               # Training logs and evaluation results
├── Dokumen/                # Documentation and reports
└── .vscode/                # VS Code configuration
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- pip or conda package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/reynaldaryansyah25/srgan-div2k.git
cd srgan-div2k
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

The project uses the following key libraries:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.1.0 | Deep learning framework |
| torchvision | ≥0.16.0 | Computer vision utilities |
| numpy | ≥1.26.0 | Numerical computations |
| pillow | ≥10.0.0 | Image processing |
| opencv-python | ≥4.8.0 | Image manipulation |
| scikit-image | ≥0.22.0 | Image metrics (PSNR, SSIM) |
| matplotlib | ≥3.8.0 | Visualization |
| tqdm | ≥4.66.0 | Progress bars |

## 🧠 Model Architecture

### Generator
- Input: Low-resolution image (LR)
- 9×9 convolutional layer for initial feature extraction
- Multiple residual blocks for deep feature learning
- Pixel shuffle layers for upsampling (4× upsampling)
- Output: Super-resolved image (SR)

### Discriminator
- Evaluates realism of generated high-resolution images
- VGG-style architecture with batch normalization
- Determines whether images are real or generated

## 📊 Training and Evaluation

### Training Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality
- **SSIM (Structural Similarity Index)**: Evaluates perceptual similarity

### Notebooks

1. **`02_training_curves.ipynb`**: 
   - Visualize training progress
   - Plot PSNR and SSIM curves across epochs
   - Monitor model convergence

2. **`03_benchmark_metrics.ipynb`**: 
   - Load trained model
   - Evaluate on test images
   - Compare metrics and visual results

## 💻 Usage

### Training the Model
```python
# Training script (typically in src/train/)
python train.py --config config.yaml
```

### Running Inference
```python
# Load pretrained model and generate super-resolution images
from src.train.model import Generator
import torch
from PIL import Image

# Load model
generator = Generator()
generator.load_state_dict(torch.load('model.pth'))

# Process image
low_res_image = Image.open('low_res.jpg')
# ... preprocessing ...
high_res_output = generator(low_res_tensor)
```

### Analyzing Results
Open and run the Jupyter notebooks:
```bash
jupyter notebook notebooks/02_training_curves.ipynb
jupyter notebook notebooks/03_benchmark_metrics.ipynb
```

## 📈 Results

Training evaluation results are saved in `outputs/logs/` with JSON format containing:
- PSNR values per epoch
- SSIM values per epoch
- Loss metrics

## 🎓 References

- **SRGAN Paper**: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
  - Authors: Ledig et al., 2017
  
- **DIV2K Dataset**: Diverse 2K resolution high-quality images
  - Link: http://www.vision.ee.ethz.ch/datasets/div2k/

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Submit pull requests
- Suggest improvements

## 📝 License

This project is open source and available for educational and research purposes.

## 📞 Contact & Resources

- **Repository**: https://github.com/reynaldaryansyah25/srgan-div2k
- **Author**: reynaldaryansyah25
- **Demo Video**: [View on Google Drive](https://drive.google.com/file/d/1R9GZkN32qEzMg8IQV8p9G_wGP5BfG8Ev/view?usp=sharing)

## 🔧 Troubleshooting

**Issue**: CUDA out of memory
- Solution: Reduce batch size in configuration

**Issue**: Slow training
- Solution: Ensure CUDA is properly installed and GPU is being utilized

**Issue**: Import errors
- Solution: Verify all dependencies are installed: `pip install -r requirements.txt`

---

**Last Updated**: January 2026 | Created for image super-resolution research and development
