from pathlib import Path

readme_text = """
# CAFES: Confidence-Aware Feature Enhancement and Selection for Robust Multimodal Emotion Recognition in Mental Health Monitoring

**A deep learning framework integrating audio, video, and text modalities with confidence-aware feature selection and hierarchical fusion for emotion recognition.**

📄 Paper submitted for publication | 🔬 Mental Health Monitoring & Suicide Risk Assessment

---

## 🔍 Overview

This repository provides the implementation of **CAFES**, a robust deep learning-based framework for **Multimodal Emotion Recognition (MMER)**.

> CAFES integrates confidence-aware soft feature selection, cross-modal attention fusion, and a composite loss function to improve emotion recognition from raw audio, facial video, and transcribed text — even under noisy and ambiguous conditions.

---

## 🌟 Key Features

- **Audio Processing**: 1D CNN + BiGRU with confidence-aware masking from raw waveforms.
- **Video Processing**: MTCNN face detection → RepVGG-CNN + Transformer Encoder.
- **Text Processing**: Pre-trained ALBERT transformer for deep semantic encoding.
- **Confidence Estimation**: Per-frame, per-channel soft selection using FFN (local) and GMM (global).
- **Fusion Mechanism**: Two-stage cross-modal attention and learnable weighted fusion.
- **Loss Function**: Focal loss + confidence alignment for improved robustness to imbalance.

---

## 📊 Results

### IEMOCAP Dataset

| Emotion | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| Angry   | 91.91%   | 83.33%    | 74.07% | 78.43%   |
| Happy   | 88.97%   | 78.65%    | 86.42% | 82.35%   |
| Neutral | 87.87%   | 80.25%    | 79.27% | 79.75%   |
| Sad     | 94.49%   | 87.04%    | 85.45% | 86.24%   |
| **Avg** | **90.81%** | **82.32%** | **81.30%** | **81.69%** |

---

### CMU-MOSEI Dataset

| Emotion  | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| Angry    | 75.62%   | 66.67%    | 65.12% | 65.88%   |
| Disgust  | 76.67%   | 75.00%    | 75.56% | 75.28%   |
| Fear     | 73.55%   | 66.67%    | 53.57% | 59.41%   |
| Happy    | 73.36%   | 47.22%    | 72.86% | 57.30%   |
| Sad      | 68.62%   | 62.35%    | 54.08% | 57.92%   |
| Surprise | 73.03%   | 60.87%    | 70.00% | 65.12%   |
| **Avg**  | **73.48%** | **63.13%** | **65.20%** | **63.48%** |

---

## 🧠 Model Architecture

- **Text**: ALBERT → CLS embedding
- **Audio**: Raw waveform → CNN → Confidence Mask → BiGRU → Temporal Attention
- **Video**: MTCNN → RepVGG → Spatial Attention + Confidence Mask → Transformer
- **Fusion**: 2-stage cross-attention (AV → AV+Text) → Weighted fusion → Softmax/Sigmoid

---

## 📁 Datasets

- **IEMOCAP**: Single-label, 4 emotions.
- **CMU-MOSEI**: Multi-label, 6 emotions.
- Processed versions: See [Download Processed Data](https://github.com/eyeness12/Confidence-Aware-Feature-Selection-for-Robust-MMER-in-Mental-Health-Monitoring)

---

## 📦 Built With

- Python 3.7+
- PyTorch
- NumPy / Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 🚀 Installation

```bash
git clone https://github.com/eyeness12/Confidence-Aware-Feature-Selection-for-Robust-MMER-in-Mental-Health-Monitoring.git
cd Confidence-Aware-Feature-Selection-for-Robust-MMER-in-Mental-Health-Monitoring
pip install -r requirements.txt
