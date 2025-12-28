# GM-DF: Generalized Multi-Scenario Deepfake Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of the paper **"GM-DF: Generalized Multi-Scenario Deepfake Detection"** (ACM MM 2025).

This framework achieves state-of-the-art cross-domain deepfake detection by combining:
- 🧠 **CLIP Vision-Language Model** as a frozen backbone
- 🔀 **Mixture of Experts (MoE)** for domain-specific adaptation
- 🎯 **Meta-Learning (MAML)** for generalization across multiple datasets
- 🖼️ **Masked Image Modeling (MIM)** for capturing fine-grained forgery artifacts

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GM-DF.git
cd GM-DF

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# OR source venv/bin/activate  # Linux/Mac

# Install dependencies (requires git)
pip install -r requirements.txt
pip install scikit-learn  # Essential for metrics (EER, AUC)
```

### Dataset Preparation

Organize your deepfake datasets in the following structure. **Note:** MAML training requires at least **two domains**.

```
datasets/
├── FaceForensics/   <-- Domain A
│   ├── real/
│   └── fake/
├── CelebDF/         <-- Domain B
│   ├── real/
│   └── fake/
└── WildDeepfake/    <-- (Optional) Test Domain
    ├── real/
    └── fake/
```

---

## 🏋️ Training

To train the model using the **Meta-Learning** strategy (training on one domain, adapting to another in each step):

```bash
# Windows
python train_meta.py --data_root datasets --domains FaceForensics CelebDF --batch_size 32 --epochs 40

# Linux/Mac
python train_meta.py --data_root ./datasets --domains FaceForensics CelebDF --batch_size 32 --epochs 40
```

> **Note:** The training uses Automatic Mixed Precision (AMP) by default for faster training on NVIDIA GPUs.

---

## 🔍 Evaluation & Testing

We provide a dedicated script `test_model.py` to evaluate your trained model on any dataset.

### Metrics
The script reports the standard metrics from the paper:
- **AUC (Area Under Curve)**
- **Accuracy**
- **EER (Equal Error Rate)**

### How to Test

**1. Self-Evaluation (Same Domain):**
Test on one of the domains used during training.
```bash
python test_model.py --data_root datasets --domains FaceForensics --model_path checkpoints/best_model.pt
```

**2. Cross-Dataset Generalization (New Domain):**
Test on a dataset the model has *never seen* (e.g., WildDeepfake) to measure generalization.
```bash
python test_model.py --data_root datasets --domains WildDeepfake --model_path checkpoints/best_model.pt
```

### Batch Inference API
You can also use the code directly in Python:

```python
import torch
from model.gm_df import build_model
from data.transforms import get_val_transforms
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(device=device)
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict
img = Image.open("test.jpg").convert("RGB")
img_tensor = get_val_transforms()(img).unsqueeze(0).to(device)
domain_id = torch.tensor([0]).to(device) # 0=FF, 1=CelebDF...

logits = model(img_tensor, domain_id)["logits"]
prob = torch.sigmoid(logits).item()
print(f"Fake Probability: {prob:.4f}")
```

---

## 💻 Platform-Specific Instructions

### Windows (NVIDIA GPU)
1. Install CUDA 11.8 or 12.1.
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Run training as shown above.

### Mac (M1/M2/M3)
The code supports Apple Silicon (MPS).
```bash
python train_meta.py --device mps --data_root ./datasets ...
```

---

## 📊 Implementation Fidelity

This implementation closely follows the paper with the following notes:
- **Architecture:** 1:1 match (CLIP, MoE, Prompt Learner, Second-Order Aggregation).
- **Training:** Uses an optimized Alternating Optimization strategy for the MAML update rule (improving stability and speed).
- **MIM:** Uses efficient random-projection tokenizer.
- **Metrics:** Reports EER, AUC, and Accuracy.

---

## 📄 License
MIT License.
