# Dementia Speech Classification Pipeline: Wav2Vec 2.0 with Stochastic Augmentation

An end-to-end machine learning pipeline that fine-tunes a Wav2Vec 2.0 transformer model to detect early signs of dementia from human speech patterns. This project engineers a dynamic, stochastic audio augmentation architecture in PyTorch to combat the severe overfitting risks associated with small, HIPAA-constrained medical datasets.

## Project Overview
Dementia is a neurodegenerative condition affecting memory and communication. Because clinical speech datasets are typically small due to privacy laws, deep learning models often overfit by memorizing baseline background noise rather than learning true acoustic markers of cognitive decline. 

This project solves this by introducing a **stochastic (dynamic) data augmentation pipeline** that alters the audio in RAM on the fly during training, forcing the transformer to learn robust phonetic features. 

## The Baseline Architecture
This project builds upon the foundational architecture of the open-source [DementiaNet repository](https://github.com/shreyasgite/dementianet). 

The original codebase was debugged, optimized, and run as a control group. Training the Wav2Vec 2.0 model on the un-augmented baseline dataset for 22 epochs resulted in a flat performance ceiling:
* **Baseline Accuracy:** 77.0%
* **Baseline Precision:** 77.0%

## Engineering Contribution: Stochastic Augmentation
To push past the baseline, I engineered a custom dynamic augmentation pipeline using the `audiomentations` library. 

Rather than generating a static folder of distorted audio files (which the model would eventually memorize), this pipeline intercepts the PyTorch dataloader. With a configurable probability threshold (p), it dynamically injects:
1. **Gaussian Noise:** Simulates variable clinical background static.
2. **Time-Stretching:** Alters the phonetic pacing (a key marker of dementia).
3. **Pitch-Shifting:** Modifies vocal depth.

### The Ablation Study (Hyperparameter Tuning)
To prevent the aggressive audio distortion from destroying the human speech signal, I conducted an ablation study across four probability environments (p=0.0, 0.25, 0.50, 0.75) to find the optimal ratio of clean-to-augmented data per epoch.

The study revealed a clear "Goldilocks zone." At p=0.25, the model underperformed (77.08%) due to memorization. At p=0.75, accuracy dropped (72.91%) because aggressive pitch-shifting masked the subtle phonetic markers of dementia. The 50/50 ratio (p=0.50) served as the optimal regularization threshold.

## Final Results
Training the model for 40 epochs utilizing the optimized stochastic pipeline (p=0.50) yielded significant improvements over the baseline:

| Metric | Baseline (Control) | Augmented Model (Optimal) |
| :--- | :--- | :--- |
| **Accuracy** | 77.0% | **81.25%** |
| **Precision** | 77.0% | **83.60%** |
| **Recall** | 77.0% | **79.10%** |
| **F1-Score** | 77.0% | **81.20%** |

*(Note: The spike in Precision to 83.6% is particularly critical for medical AI, as minimizing false positives reduces severe psychological distress for healthy patients.)*

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
└── README.md
```

## Future Enhancements
* **Ensemble Modeling:** Combining the Wav2Vec acoustic analysis with a CNN-LSTM network trained on textual transcripts for multi-modal diagnosis.
* **Alternative Transformers:** Testing the augmentation pipeline against the HuBERT architecture for potentially stronger high-noise feature extraction.
* **Dynamic Probability Scaling:** Implementing a system where the intensity of the noise augmentation automatically adjusts based on real-time validation loss spikes during training.
