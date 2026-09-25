# ModelForge

> **Transparent, local-first AutoML experimentation engine for automated machine learning pipeline discovery.**

ModelForge is a Python-based AutoML framework designed to make machine learning experimentation **automated, reproducible, inspectable, and local-first**.

Instead of hiding the entire machine learning workflow behind a single black-box function, ModelForge exposes the major stages of the ML lifecycle:

```text
Dataset
   ↓
Target Selection
   ↓
Dataset Profiling
   ↓
Column Intelligence
   ↓
Data Quality & Leakage Audit
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Preprocessing
   ↓
Model Registry
   ↓
Pipeline Generation
   ↓
Model Screening
   ↓
Cross-Validation
   ↓
Evaluation
   ↓
Ranking
   ↓
Explainability
   ↓
Persistence
   ↓
Prediction
