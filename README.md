# AI-Based Bird Species Classification and Explainable Image Analysis

A deep learning-based bird image classification system that detects birds, classifies them into 12 supported Indian bird species, and provides visual explanations using Grad-CAM.

The project combines binary bird/non-bird detection, transfer learning with MobileNetV2, and an interactive Streamlit web application.

---

## Project Overview

Bird species identification from images can be challenging because bird appearance varies with:

- Pose
- Lighting
- Background
- Camera quality
- Viewing angle
- Similar visual characteristics between species

This project develops an end-to-end deep learning pipeline to address the classification problem.

The system performs the following stages:

```text
Input Image
     │
     ▼
Bird / Non-Bird Detection
     │
     ├── Non-Bird → Reject
     │
     └── Bird
           │
           ▼
     12-Class Bird Classification
           │
           ▼
      Top-3 Predictions
           │
           ▼
       Grad-CAM
           │
           ▼
   Visual Explanation