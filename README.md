# Dementia Speech Classification Pipeline: Wav2Vec 2.0 with Stochastic Augmentation

[cite_start]An end-to-end machine learning pipeline that fine-tunes a Wav2Vec 2.0 transformer model to detect early signs of dementia from human speech patterns[cite: 10, 23]. [cite_start]This project engineers a dynamic, stochastic audio augmentation architecture in PyTorch to combat the severe overfitting risks associated with small, HIPAA-constrained medical datasets[cite: 47, 48].

## Project Overview
[cite_start]Dementia is a neurodegenerative condition affecting memory and communication[cite: 7]. [cite_start]Because clinical speech datasets are typically small due to privacy laws, deep learning models often overfit by memorizing baseline background noise rather than learning true acoustic markers of cognitive decline[cite: 48]. 

[cite_start]This project solves this by introducing a **stochastic (dynamic) data augmentation pipeline** that alters the audio in RAM on the fly during training, forcing the transformer to learn robust phonetic features[cite: 49]. 

## The Baseline Architecture
[cite_start]This project builds upon the foundational architecture of the open-source [DementiaNet repository](https://github.com/shreyasgite/dementianet)[cite: 291, 292]. 

[cite_start]The original codebase was debugged, optimized, and run as a control group[cite: 51]. Training the Wav2Vec 2.0 model on the un-augmented baseline dataset for 22 epochs resulted in a flat performance ceiling:
* [cite_start]**Baseline Accuracy:** 77.0% [cite: 74]
* [cite_start]**Baseline Precision:** 77.0% [cite: 74]

## Engineering Contribution: Stochastic Augmentation
[cite_start]To push past the baseline, I engineered a custom dynamic augmentation pipeline using the `audiomentations` library[cite: 53]. 

Rather than generating a static folder of distorted audio files (which the model would eventually memorize), this pipeline intercepts the PyTorch dataloader. With a configurable probability threshold ($p$), it dynamically injects:
1. [cite_start]**Gaussian Noise:** Simulates variable clinical background static[cite: 47].
2. [cite_start]**Time-Stretching:** Alters the phonetic pacing (a key marker of dementia)[cite: 47, 274].
3. [cite_start]**Pitch-Shifting:** Modifies vocal depth[cite: 47, 267].

### The Ablation Study (Hyperparameter Tuning)
[cite_start]To prevent the aggressive audio distortion from destroying the human speech signal, I conducted an ablation study across four probability environments ($p=0.0, 0.25, 0.50, 0.75$)[cite: 245, 246]. 

![Ablation Study](assets/ablation_study_graph.png) 
*(Insert your ablation study graph here)*

The study revealed a clear "Goldilocks zone." [cite_start]At $p=0.25$, the model underperformed (77.08%) due to memorization[cite: 257]. [cite_start]At $p=0.75$, accuracy dropped (72.91%) because aggressive pitch-shifting masked the subtle phonetic markers of dementia[cite: 258]. [cite_start]The 50/50 ratio ($p=0.50$) served as the optimal regularization threshold[cite: 259].

## Final Results
[cite_start]Training the model for 40 epochs utilizing the optimized stochastic pipeline ($p=0.50$) yielded significant improvements over the baseline[cite: 53, 106]:

| Metric | Baseline (Control) | Augmented Model (Optimal) |
| :--- | :--- | :--- |
| **Accuracy** | 77.0% | **81.25%** |
| **Precision** | 77.0% | **83.60%** |
| **Recall** | 77.0% | **79.10%** |
| **F1-Score** | 77.0% | **81.20%** |

[cite_start]*(Note: The spike in Precision to 83.6% is particularly critical for medical AI, as minimizing false positives reduces severe psychological distress for healthy patients[cite: 78, 90].)*

## Repository Structure

```text
DementiaNet-Augmented/
│
├── notebooks/
│   ├── 01_baseline_wav2vec.ipynb     # The debugged original architecture (Control)
│   └── 02_stochastic_pipeline.ipynb  # The augmented pipeline (p=0.50)
│
├── reports/
│   ├── Final_Report_NNDL.pdf         # Comprehensive academic breakdown & Error Analysis
│   └── Presentation_Slides.pdf       # High-level architecture overview
│
├── assets/                           # Graphs, ROC curves, and Confusion Matrices
│
└── README.md
