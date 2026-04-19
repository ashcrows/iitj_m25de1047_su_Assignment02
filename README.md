# Speech Understanding — Programming Assignment 2
**IIT Jodhpur | MTech Data Engineering | M25DE1047 | Spring 2026**

---

## What This Does

This project builds a complete pipeline that takes a Hindi-English (Hinglish) classroom lecture recording and:

1. **Transcribes it** with frame-level language detection and constrained Whisper decoding
2. **Converts it to IPA** using a custom Hinglish phoneme mapper
3. **Translates it to Maithili** using a 500-entry parallel corpus
4. **Synthesizes it in Maithili** using your own voice (zero-shot cloning)
5. **Tests robustness** with an anti-spoofing classifier and adversarial attacks

---

## Project Structure

```
.
├── pipeline.py              ← Run this (full pipeline)
├── prepare_data.py          ← Run first (converts M4A → WAV)
├── evaluate.py              ← Generate metrics report
├── setup_check.py           ← Verify your environment
├── install.sh               ← One-command install (macOS)
├── requirements.txt
├── configs/
│   └── default.yaml         ← All hyperparameters
├── src/
│   ├── part1/
│   │   ├── denoising.py         # Spectral subtraction denoiser
│   │   ├── lid_model.py         # CNN + MHA + BiLSTM frame-level LID
│   │   └── constrained_asr.py   # Whisper + N-gram logit biasing
│   ├── part2/
│   │   ├── ipa_mapper.py        # Hinglish G2P → IPA
│   │   └── lrl_translator.py    # Maithili parallel corpus translator
│   ├── part3/
│   │   ├── voice_embedding.py   # TDNN x-vector speaker embedding
│   │   ├── prosody_warping.py   # PYIN F0 + DTW prosody transfer
│   │   └── tts_synthesizer.py   # VITS / MMS synthesis
│   ├── part4/
│   │   ├── anti_spoofing.py     # LFCC + LCNN countermeasure
│   │   └── adversarial.py       # FGSM mel-domain attack
│   └── utils/
│       ├── audio_utils.py       # Shared audio I/O, mel, LFCC, SNR
│       └── evaluation.py        # WER, MCD, EER, confusion matrix
├── tests/
│   └── test_pipeline.py         # 26 unit tests
├── data/                        ← Created by prepare_data.py
├── outputs/                     ← Created by pipeline.py
├── report.tex                   ← LaTeX source for submission report
└── implementation_note.md       ← Required 1-page notes per question
```

---

## Setup (macOS Apple Silicon)

### Step 1 — System dependencies
```bash
brew install ffmpeg espeak-ng
```

### Step 2 — Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **Important:** After activating, make sure `python3` points to the venv:
> ```bash
> which python3   # should show .venv/bin/python3
> ```
> If it still shows `/opt/homebrew/bin/python3`, run:
> ```bash
> unalias python3 && unalias python
> ```

### Step 3 — Install packages
```bash
# Recommended: one command
chmod +x install.sh && ./install.sh

# Or manually:
pip install -r requirements.txt
```

### Step 4 — Verify setup
```bash
python3 setup_check.py
```

---

## Running the Pipeline

### Prepare data first
```bash
python3 prepare_data.py \
    --input_m4a data/Main.m4a \
    --voice_ref data/voice_test_add.wav
```

> Record 60 seconds of your own voice and use it as `--voice_ref` for best synthesis quality.

### Run full pipeline
```bash
python3 pipeline.py \
    --input_audio data/original_segment.wav \
    --voice_ref data/student_voice_ref.wav \
    --target_lrl maithili \
    --device mps
```

> Use `--device mps` on Apple Silicon, `--device cuda` on NVIDIA GPU, `--device cpu` as fallback.

### Run individual parts
```bash
python3 pipeline.py --run_part part1   # STT only
python3 pipeline.py --run_part part2   # IPA + translation (needs part1)
python3 pipeline.py --run_part part3   # Voice cloning (needs parts 1+2)
python3 pipeline.py --run_part part4   # Anti-spoofing (needs parts 1+3)
```

### Generate evaluation report
```bash
python3 evaluate.py \
    --transcript outputs/transcript.json \
    --synth_audio outputs/output_LRL_cloned.wav \
    --voice_ref data/student_voice_ref.wav
```

### Run tests
```bash
python3 tests/test_pipeline.py
```

---

## What Each Part Does

### Part I — Code-Switched Transcription

**Task 1.3 (Denoising):** Spectral subtraction removes classroom background noise. Estimates noise from the first 200ms, then subtracts it with over-subtraction factor α=2.0 and spectral floor β=0.002 to prevent musical noise artifacts.

**Task 1.1 (LID):** A CNN + Multi-Head Attention + BiLSTM model classifies each 10ms frame as English or Hindi. A 5-frame median filter smooths predictions before detecting language switch timestamps. Target: F1 ≥ 0.85 per language.

**Task 1.2 (Constrained ASR):** Whisper-large-v3 with two modifications:
- Per-chunk language forcing based on LID results (`language="en"` or `"hi"`)  
- N-gram logit biasing (+2.5 log-prob) for technical terms from the course syllabus

### Part II — Phonetic Mapping and Translation

**Task 2.1 (IPA):** Custom Hinglish G2P with 4-tier priority: Devanagari rules → Romanized Hindi lookup → English academic dictionary → espeak-ng fallback. Handles retroflex consonants (ʈ, ɖ) that standard G2P tools miss.

**Task 2.2 (Translation):** Word-level translation to Maithili using a 144-entry bundled corpus (extensible to 500+ via `data/maithili_corpus.json`). Technical terms are phonetically borrowed unchanged — linguistically authentic for academic register.

### Part III — Voice Cloning

**Task 3.1 (Speaker Embedding):** TDNN x-vector architecture extracts a 512-d speaker embedding from your 60s voice reference. Statistics pooling (mean + std) aggregates frame-level features to utterance level.

**Task 3.2 (Prosody Warping):** PYIN extracts F0 (fundamental frequency) from the professor's lecture. DTW alignment in log-Hz domain maps prosodic shape onto synthesized speech — log domain because musical intervals are logarithmic.

**Task 3.3 (Synthesis):** Auto-selects TTS backend: coqui-tts YourTTS → Meta MMS-TTS → TorchAudio Tacotron2 → mock. Output at 22.05kHz with 50ms cross-fade between segments.

### Part IV — Anti-Spoofing and Adversarial Robustness

**Task 4.1 (CM):** LFCC features (60 coefficients, linear filterbank) fed into a Light CNN with Max Feature Map activations. Evaluated by Equal Error Rate (EER). LFCC preferred over MFCC because synthesized speech artifacts cluster at linear frequency bands.

**Task 4.2 (FGSM):** Fast Gradient Sign Method applied in mel-spectrogram domain. Binary search finds the minimum ε that flips LID Hindi→English while keeping SNR > 40dB (inaudible). Waveform reconstructed via Griffin-Lim.

---

## Evaluation Criteria

| Metric | Threshold | Where computed |
|---|---|---|
| WER (English segments) | < 15% | `evaluate.py` |
| WER (Hindi segments) | < 25% | `evaluate.py` |
| MCD (synthesized voice) | < 8.0 dB | `evaluate.py` |
| LID F1 (per language) | ≥ 0.85 | `src/utils/evaluation.py` |
| Switch timestamp precision | ± 200ms | `src/utils/evaluation.py` |
| Anti-spoofing EER | < 10% | `src/part4/anti_spoofing.py` |
| FGSM epsilon (SNR > 40dB) | reported | `src/part4/adversarial.py` |
| Output sample rate | ≥ 22.05kHz | `src/part3/tts_synthesizer.py` |

---

## Output Files

| File | Description |
|---|---|
| `outputs/denoised_segment.wav` | Cleaned lecture audio |
| `outputs/lid_results.json` | Per-frame language labels + switch timestamps |
| `outputs/transcript.json` | Full transcript with segment timestamps |
| `outputs/ipa_transcript.json` | IPA representation of each segment |
| `outputs/lrl_translation_maithili.json` | Maithili translation |
| `outputs/speaker_embedding.pt` | Your 512-d x-vector |
| `outputs/warped_prosody.pt` | F0 + energy features |
| `outputs/output_LRL_cloned.wav` | **Final synthesized lecture (22.05kHz)** |
| `outputs/cm_evaluation.json` | Anti-spoofing EER results |
| `outputs/adversarial_results.json` | FGSM epsilon + SNR table |
| `outputs/evaluation_report.json` | Full metrics summary |

---

## References

1. Snyder et al. (2018). X-vectors: Robust DNN embeddings for speaker recognition. *ICASSP*.
2. Radford et al. (2023). Robust speech recognition via large-scale weak supervision. *ICML*. (Whisper)
3. Kim et al. (2021). VITS: Conditional variational autoencoder with adversarial learning for TTS. *ICML*.
4. Wu et al. (2020). Light CNN for deep fake detection. *INTERSPEECH*.
5. Goodfellow et al. (2015). Explaining and harnessing adversarial examples. *ICLR*.
6. Pratap et al. (2023). Scaling speech technology to 1,000+ languages. *ACL*. (Meta MMS)
7. Mauch & Dixon (2014). PYIN: A fundamental frequency estimator. *ICASSP*.
8. Boll (1979). Suppression of acoustic noise via spectral subtraction. *IEEE Trans. Acoustics*.

---

*Student: Ashish Sinha | M25DE1047 | IIT Jodhpur | Speech Understanding | Spring 2026*
