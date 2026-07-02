# Fracture Detection for Radiology Triage

> AI-powered computer vision pipeline for automatic fracture detection and localization on X-ray images.

## Overview

This project aims to develop a deep learning pipeline capable of automatically detecting bone abnormalities on conventional radiographs.

The primary objective is **radiology triage**: prioritize examinations that are likely to contain fractures, allowing radiologists to review urgent cases first.

The first version of the project focuses exclusively on the **computer vision pipeline**, from medical image preprocessing to explainable object detection.

---

## Pipeline

```text
X-ray
   │
   ▼
Medical Image Denoising
(U-Net + NAFNet Blocks)
   │
   ▼
Enhanced Radiograph
   │
   ▼
YOLO Object Detection
(Localize Fracture)
   │
   ▼
Pathology Classification
   │
   ▼
Grad-CAM Explainability
   │
   ▼
Prediction + Visualization
```

---

## Features

- Medical image denoising
- Fracture localization using YOLO
- Pathology classification
- Explainable AI with Grad-CAM
- Modular PyTorch implementation
- Experiment tracking with Weights & Biases
- CI/CD using GitHub Actions

---

## Model Architecture

### 1. Medical Image Enhancement

Radiographs are first processed by a denoising network based on:

- U-Net architecture
- NAFNet residual blocks

This preprocessing step aims to improve image quality while preserving clinically relevant anatomical structures.

---

### 2. Fracture Detection

The enhanced image is then processed using **Ultralytics YOLO** for object detection.

The detector predicts:

- Bounding boxes
- Confidence score
- Fracture category

---

### 3. Explainability

To improve model interpretability, Grad-CAM heatmaps are generated to highlight the image regions contributing to each prediction.

This allows clinicians to visually inspect where the model focuses before validating its output.

---

## Dataset

Current dataset:

- **VinDr-SpineXR**

The dataset contains musculoskeletal X-ray images with expert annotations for abnormal findings.

Future versions may include additional public fracture datasets.

---

## Project Structure

```text
fracture-detection/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── webdataset/
│
├── datasets/
│
├── models/
│   ├── denoiser/
│   ├── detector/
│   └── classifier/
│
├── training/
│
├── inference/
│
├── explainability/
│
├── notebooks/
│
├── scripts/
│
├── tests/
│
├── outputs/
│
├── requirements.txt
└── README.md
```

---

## Technologies

| Component | Technology |
|------------|------------|
| Deep Learning | PyTorch |
| Detection | Ultralytics YOLO |
| Denoising | U-Net + NAFNet |
| Explainability | Grad-CAM |
| Dataset | WebDataset |
| Experiment Tracking | Weights & Biases |
| CI/CD | GitHub Actions |
| Visualization | Matplotlib, OpenCV |

---

## Training Pipeline

```text
Raw Dataset
      │
      ▼
Preprocessing
      │
      ▼
WebDataset
      │
      ▼
Denoising Network
      │
      ▼
YOLO Training
      │
      ▼
Validation
      │
      ▼
Evaluation
      │
      ▼
Grad-CAM
```

---

## Evaluation Metrics

The object detector will be evaluated using:

- mAP@50
- mAP@50-95
- Precision
- Recall
- F1-score

Additional metrics for the denoising model include:

- PSNR
- SSIM

---

## Experiment Tracking

Training experiments are monitored with **Weights & Biases**.

Tracked information includes:

- Training and validation losses
- Detection metrics
- Learning rate
- Model checkpoints
- Prediction examples
- Grad-CAM visualizations

---

## Continuous Integration

GitHub Actions automatically performs:

- Code formatting
- Linting
- Unit tests
- Training configuration validation
- Model export verification

---

## Roadmap

### Phase 1 — Computer Vision

- [ ] Medical image denoising
- [ ] WebDataset pipeline
- [ ] YOLO training
- [ ] Fracture localization
- [ ] Pathology classification
- [ ] Grad-CAM integration
- [ ] Benchmark evaluation
- [ ] Docker support

### Phase 2 — AI Assistant

- [ ] Report generation with LLM
- [ ] Retrieval-Augmented Generation (RAG)
- [ ] Radiology report assistant

### Phase 3 — Clinical Workflow

- [ ] SQL database
- [ ] Triage dashboard
- [ ] PACS integration
- [ ] REST API
- [ ] User interface

---

## Future Work

The long-term objective is to build a complete AI-assisted radiology workflow by integrating:

- automatic fracture detection,
- report generation using Large Language Models,
- intelligent triage,
- and conversational querying of radiology cases.

---

## Disclaimer

This project is intended for research and educational purposes only. It is **not** a certified medical device and must not be used for clinical decision-making without appropriate regulatory approval and expert validation.