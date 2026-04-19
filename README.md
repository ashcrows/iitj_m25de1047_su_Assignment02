# Speech Understanding – Programming Assignment 2

**Ashish Sinha | M25DE1047**
**IIT Jodhpur | MTech Data Engineering | Spring 2026**

---

## Overview

This project implements an **end-to-end speech processing pipeline** for handling **code-switched (Hinglish) lecture audio** and converting it into a **Low Resource Language (Maithili)** using **my own voice via zero-shot voice cloning**.

Unlike standard pipelines, this implementation focuses on:

* Frame-level language understanding
* Custom decoding logic for ASR
* Manual phonetic mapping for Hinglish
* Prosody-aware voice synthesis
* Robustness against spoofing and adversarial noise

The goal was to **build the system from an architectural perspective**, not just use pre-built APIs.

---

## Pipeline Summary

The system processes the lecture audio in 4 stages:

### 1. Speech-to-Text (Part I)

* Noise removal using spectral subtraction
* Frame-level Language Identification (English vs Hindi)
* Whisper-based transcription with **custom constrained decoding**

  * Used N-gram biasing to prioritize technical words

### 2. Phonetic Mapping & Translation (Part II)

* Hinglish text converted to **IPA representation**
* Custom rule-based + dictionary mapping implemented
* Translated to Maithili using a **manually curated corpus**

### 3. Voice Cloning (Part III)

* Extracted speaker embedding from my 60-second voice sample
* Applied **prosody transfer using DTW**
* Generated final speech using TTS model

### 4. Robustness & Security (Part IV)

* Built anti-spoofing classifier using LFCC features
* Implemented FGSM adversarial attack to test LID robustness

---

## Key Design Decisions

Some important implementation choices:

* Used **frame-level LID instead of segment-level** to capture fast code-switching
* Applied **logit bias instead of full LM decoding** for better control over Whisper
* Designed a **hybrid IPA mapper** (rule-based + fallback)
* Used **log-frequency domain for DTW** (more stable for pitch alignment)
* Selected **LFCC instead of MFCC** for spoof detection (better for synthetic signals)

---

## Project Structure

```
.
├── pipeline.py
├── prepare_data.py
├── evaluate.py
├── requirements.txt
├── src/
│   ├── part1/   # STT + LID + Denoising
│   ├── part2/   # IPA + Translation
│   ├── part3/   # Voice Cloning
│   ├── part4/   # Anti-spoofing + Adversarial
│   └── utils/
├── data/
├── outputs/
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install system tools

```bash
brew install ffmpeg espeak-ng
```

---

## Running the Pipeline

### Step 1: Prepare data

```bash
python prepare_data.py
```

### Step 2: Run full pipeline

```bash
python pipeline.py
```

### Step 3: Evaluate results

```bash
python evaluate.py
```

---

## Outputs

The pipeline generates:

* Transcribed Hinglish text
* IPA representation
* Maithili translation
* Final synthesized audio (22.05 kHz)
* Evaluation metrics (WER, MCD, EER)

---

## Evaluation Metrics

The system is evaluated based on:

* Word Error Rate (WER)
* Mel Cepstral Distortion (MCD)
* LID Accuracy
* Equal Error Rate (EER)
* Adversarial robustness (epsilon vs SNR)

---

## Limitations

* Hinglish IPA mapping is partially rule-based and may miss rare patterns
* Translation quality depends on size of custom corpus
* TTS quality varies based on backend availability

---

## References

* Whisper (OpenAI)
* VITS / YourTTS
* X-vector speaker embedding
* PYIN pitch extraction
* FGSM adversarial method

---

## Author

Ashish Sinha
M25DE1047
MTech Data Engineering
IIT Jodhpur

---

*This implementation was developed as part of the Speech Understanding course assignment with focus on system-level integration and custom modeling.*
