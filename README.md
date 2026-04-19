# Speech Understanding — Programming Assignment 2

**IIT Jodhpur | MTech Data Engineering | M25DE1047 | Spring 2026**

---

## Overview

This project implements an **end-to-end speech processing pipeline** for handling **code-switched (Hinglish) lecture audio** and converting it into a **Low Resource Language (Maithili)** using **zero-shot voice cloning with my own voice**.

The pipeline integrates:

* Speech-to-Text (STT) with language-aware decoding
* Phonetic normalization using IPA
* Translation to a low-resource language
* Voice cloning with prosody transfer
* Anti-spoofing and adversarial robustness

The focus of this assignment is on **system-level integration and custom logic implementation**, rather than direct usage of APIs.

---

## ⚠️ Why Notebook (.ipynb) is Used

This project is implemented using a **Jupyter Notebook (.ipynb)** instead of standalone scripts due to the following reasons:

### 1. GPU Requirement

Several components require **GPU acceleration**:

* Whisper / ASR models
* Speaker embedding extraction
* TTS synthesis models
* Adversarial perturbation generation

Local systems (especially CPU-only or Mac environments) are **not sufficient** to run the full pipeline efficiently.

---

### 2. Colab / Kaggle Compatibility

The notebook is designed to run on:

* **Google Colab (recommended)**
* **Kaggle Notebook (T4 / GPU100 environments)**

These platforms provide:

* Free GPU (T4 / A100 depending on availability)
* Pre-installed deep learning libraries
* Faster execution for model inference

---

### 3. Step-by-Step Execution

The notebook structure allows:

* Running each stage independently
* Debugging intermediate outputs
* Visualizing results (spectrograms, outputs, etc.)

This is especially useful for:

* Multi-stage pipelines (Part I → IV)
* Testing individual components (LID, TTS, DTW, etc.)

---

### 4. Reproducibility

Using a notebook ensures:

* Controlled execution environment
* Clear sequence of steps
* Easy reproducibility for evaluation

---

## 🧠 Pipeline Summary

### Part I — Code-Switched Transcription

* Denoising using spectral subtraction
* Frame-level Language Identification (English vs Hindi)
* Whisper-based transcription with constrained decoding

### Part II — Phonetic Mapping & Translation

* Hinglish → IPA conversion using custom rules
* Translation to Maithili using curated dictionary

### Part III — Voice Cloning

* Speaker embedding from 60s voice sample
* Prosody transfer using DTW (F0 + energy)
* Speech synthesis using TTS model

### Part IV — Robustness

* Anti-spoofing classifier using LFCC features
* FGSM-based adversarial attack on LID

---

## 🚀 How to Run (Recommended)

### Option 1 — Google Colab (Best)

1. Open the notebook in Colab
2. Enable GPU:

   * Runtime → Change runtime type → GPU
3. Run all cells sequentially

---

### Option 2 — Kaggle Notebook

1. Upload notebook + dataset
2. Enable GPU (T4 / GPU100)
3. Run cells

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

Additional system dependencies:

```bash
apt-get install ffmpeg espeak-ng
```

---

## 📁 Inputs

* `Main.m4a` → Lecture audio
* `voice_test_add.wav` → 60-second voice sample

---

## 📁 Outputs

* Transcription (Hinglish)
* IPA representation
* Maithili translation
* Final synthesized audio (22.05 kHz)
* Evaluation metrics (WER, MCD, EER)

---

## 📊 Evaluation Metrics

* Word Error Rate (WER)
* Mel Cepstral Distortion (MCD)
* LID Accuracy
* Equal Error Rate (EER)
* Adversarial robustness (epsilon vs SNR)

---

## ⚠️ Limitations

* Hinglish IPA mapping is partially rule-based
* Translation quality depends on dictionary coverage
* GPU availability may vary across platforms

---

## Author

Ashish Sinha
M25DE1047
MTech Data Engineering
IIT Jodhpur
