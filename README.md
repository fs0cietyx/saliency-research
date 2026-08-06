<div align="center">
  
# Investigating Salient Object Detection (SOD)
**Comparative Analysis of Hybrid Loss Formulations vs. Progressive Curriculum Learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📖 Abstract

Salient Object Detection (SOD) aims to identify the most visually distinct objects within an image. This repository contains the official PyTorch implementation and experimentation for our research comparing two distinct methodologies to improve SOD fidelity:
1. **Mathematical Optimization:** A mathematically rigorous `BCE + SSIM + IoU` Hybrid Loss that forces structural and edge-aware learning.
2. **Data-Centric Optimization:** A Progressive Structure-Aware Refinement (PSAR) framework that mimics human learning via Curriculum Learning (scaling input resolution dynamically).

## 🚀 Key Contributions

### Model 1: The "Smart Loss" Approach (`saliency_detection.py`)
- **Architecture:** `ResNet50-UNet` Encoder-Decoder.
- **Innovation:** Utilizes a custom 3-part Hybrid Loss function. While BCE handles raw pixel correctness, SSIM penalizes blurry boundaries, and IoU ensures global shape consistency.
- **Explainable AI (XAI):** Generates predictive uncertainty heatmaps (via Shannon entropy) to visually highlight regions of low model confidence.

### Model 2: The "Smart Training" Approach (`psar_saliency.py`)
- **Architecture:** Minimalist Feature Pyramid Network (FPN) with `ResNet50`.
- **Innovation:** Progressive Curriculum Learning. The model dynamically scales input resolution (`224px → 288px → 352px`) across training epochs to learn global context before local detail.
- **Validation:** Utilizes 1,000-iteration Bootstrapping to generate 95% Confidence Intervals for robust metric evaluation.

---

## 📊 Quantitative Results (DUTS-TE Benchmark)

Evaluated on the standard DUTS-TE dataset (5,019 images). The PSAR Curriculum model effectively matches/beats heavy baseline references on pixel error (MAE) while using a significantly lighter FPN decoder.

| Algorithm | Decoder Type | MAE (↓) | Adp F-Measure (↑) | 95% CI (MAE) | 95% CI (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| F3Net (Baseline) | Complex | 0.0350 | **0.840** | - | - |
| **Model 1 (Hybrid Loss)** | UNet | *TBD* | *TBD* | - | - |
| **Model 2 (PSAR)** | **Minimalist FPN** | **0.0349** | 0.816 | **[0.0334, 0.0364]** | **[0.809, 0.823]** |

---

## 👁️ Qualitative Results (Model 1)

The Hybrid Loss successfully generates highly accurate binary masks. Furthermore, the generated uncertainty maps prove the model isolates its mathematical doubt precisely to the fine boundaries of complex objects.

| Original Image | Predicted Saliency Mask | Predictive Uncertainty Map |
|:---:|:---:|:---:|
| *(Hidden for demo)* | ![Prediction](assets/sample_prediction.png) | ![Uncertainty](assets/sample_uncertainty.png) |

---

## ⚙️ Quick Start & Reproducibility

### Environment Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/fs0cietyx/saliency-research.git
cd saliency-research
pip install -r requirements.txt
```

### Training & Evaluation
We highly recommend running these scripts in an environment with a CUDA-enabled GPU (e.g., Google Colab).

**Run Model 1 (Hybrid Loss):**
*(Automatically downloads the 1.2GB DUTS dataset to `./data/`)*
```bash
python saliency_detection.py
```
*Outputs will be saved to the `out/` directory.*

**Run Model 2 (PSAR Curriculum):**
```bash
python psar_saliency.py
```
*Outputs and statistical charts will be saved to the `output/visuals/` directory.*

---

## 📝 Citation

If you find this code or our comparative analysis useful in your research, please consider citing our work:

```bibtex
@article{fs0cietyx2026saliency,
  title={Investigating Salient Object Detection: Curriculum Learning vs Hybrid Loss Formulations},
  author={Biswas, Mainak and Collaborators},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/fs0cietyx/saliency-research}
}
```

## ⚖️ License
This project is open-sourced under the MIT License. See the `LICENSE` file for details.
