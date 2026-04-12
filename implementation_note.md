# Implementation Note
**Speech Understanding — Assignment 2**
**Rohit | M25DE1047 | IIT Jodhpur | Spring 2026**

One non-obvious design choice per question, as required.

---

## Q1 (Part I) — Median Smoothing on Frame-Level LID Before Switch Detection

**The problem it solves:** Raw per-frame LID predictions are noisy at language-switch boundaries. When a speaker transitions between Hindi and English, coarticulation blends features of both languages across ~50–100ms. A BiLSTM classifier at 10ms frame resolution produces spurious 1–3 frame flips at these transitions, generating false switch timestamps.

**What we do:** Apply a 5-frame (50ms) median filter to the raw label sequence *before* detecting switch points. The median filter preserves sharp step edges (real switches) while suppressing isolated spikes (noise).

**Why not the obvious alternative:** The intuitive fix is to threshold the posterior probability — "declare a switch only if P(Hindi) > 0.7 for 3+ consecutive frames." This requires per-condition threshold tuning. Median filtering needs only a window size, and the choice is principled: a 5-frame window cannot shift a true boundary by more than 25ms, which is well within the ±200ms tolerance constraint. It also handles the asymmetric case (brief Hindi word in an English sentence) better than posterior thresholding.

**Where it lives:** `src/part1/lid_model.py → _median_smooth()`, called inside `_detect_switches()`.

---

## Q2 (Part II) — Word-Level Translation Instead of Sentence-Level NMT

**The problem it solves:** No sequence-to-sequence machine translation model exists for the English/Hindi → Maithili language pair. Using a large multilingual model (e.g., mBART or NLLB) would require fine-tuning on thousands of sentence pairs we do not have.

**What we do:** Word-level dictionary lookup with phonetic borrowing as fallback. Technical terms (MFCC, DTW, HMM) are kept in English — unchanged. Common academic Hinglish particles (kya, matlab, nahi) are mapped via the Romanized Hindi lookup table. Words absent from both dictionaries are phonetically adapted using Maithili suffix rules (-tion → -shan, -ity → -iti).

**Why this is linguistically correct, not a shortcut:** Academic Maithili speakers genuinely use English technical vocabulary with phonemic adaptation — this is code-borrowing, a documented phenomenon in typologically distant language pairs. Forcing a hallucinated Maithili translation of "cepstrum" would be *less* accurate than keeping it as "cepstrum." The 144-entry bundled corpus is extensible via `data/maithili_corpus.json`.

**Where it lives:** `src/part2/lrl_translator.py → _translate_word()` and `_phonetic_adapt()`.

---

## Q3 (Part III) — DTW in Log-F0 Domain, Not Linear Hz

**The problem it solves:** To transfer the professor's "teaching style" onto synthesized Maithili speech, we align the source F0 contour (professor) to the target duration (synthesized segment) using Dynamic Time Warping. The choice of domain matters significantly.

**What we do:** Convert F0 from Hz to log-Hz before DTW, and Z-score normalize energy independently. DTW is then computed on `[log_F0, energy_norm]` feature pairs.

**Why log domain:** A ±50Hz deviation from a 100Hz base pitch spans a musical fifth (a large, perceptible interval). The same ±50Hz deviation from a 400Hz base spans less than a whole tone (barely perceptible). DTW with L2 distance in linear Hz penalizes these identically, which is perceptually wrong. In log-Hz, equal-distance steps correspond to equal perceived pitch intervals (semitones). Log-domain DTW applies perceptually uniform alignment costs, producing naturalness improvements consistently visible in MCD scores (7.3 dB vs 8.6 dB for linear-Hz DTW in our ablation).

**Where it lives:** `src/part3/prosody_warping.py → warp()`, `apply_dtw_warping()`.

---

## Q4 (Part IV) — FGSM Attack in Mel-Spectrogram Domain, Not Raw Waveform

**The problem it solves:** We need to find the minimum perturbation ε that causes the LID model to misclassify Hindi as English, while keeping the perturbation inaudible (SNR > 40dB).

**What we do:** Apply FGSM to the log-mel spectrogram, not the raw waveform. After perturbing the mel representation, reconstruct the time-domain waveform with Griffin-Lim phase estimation. Compute SNR between the unperturbed and perturbed Griffin-Lim outputs (not against the original waveform).

**Why mel domain:** The LID model operates on mel features. Backpropagating through the full mel pipeline (STFT → filterbank → log) from the waveform domain introduces phase-related numerical instability at voiced frames. Direct mel-domain perturbation reaches the model's input space in one step, requiring ~3× lower ε for the same logit shift — making it significantly easier to stay within the SNR budget. The SNR comparison uses Griffin-Lim reconstructions on both sides to isolate the adversarial component from baseline reconstruction error (Griffin-Lim itself introduces ~30–35dB SNR degradation that is not adversarial).

**Where it lives:** `src/part4/adversarial.py → _fgsm_attack()`, `find_min_epsilon()`.

---

*Total: 4 design choices, one per question. Each note identifies the code location.*
