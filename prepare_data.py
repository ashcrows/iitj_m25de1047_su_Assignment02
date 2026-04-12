"""
Data preparation script.
Converts the provided M4A lecture recording to WAV and sets up data directory.
Run this FIRST before pipeline.py.

Usage:
    python prepare_data.py --input_m4a path/to/Recording.m4a
"""

import argparse
import logging
import os
import json
import subprocess
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prepare_data")

SPEECH_SYLLABUS = """
Speech Understanding Course Syllabus
IIT Jodhpur - Spring 2026

Module 1: Fundamentals of Speech Signals
- Digital signal processing for speech: sampling, quantization, Fourier transform
- Short-time spectral analysis: STFT, spectrogram, filterbanks
- Mel-frequency cepstral coefficients (MFCC): derivation and computation
- Cepstrum, cepstral analysis, cepstral mean normalization
- Perceptual Linear Prediction (PLP), RASTA-PLP
- Pitch (fundamental frequency) estimation: autocorrelation, YIN, PYIN
- Voiced/unvoiced decision, voice activity detection (VAD)
- Prosody: intonation, rhythm, stress, duration, energy contours

Module 2: Acoustic Phonetics
- Articulatory phonetics: vocal tract, articulators, manner/place of articulation
- Phonemes, allophones, phonological rules
- Vowels, consonants: stops, fricatives, affricates, nasals, approximants
- Coarticulation, assimilation, reduction
- Formants, formant transitions, spectral envelope
- International Phonetic Alphabet (IPA)
- Grapheme-to-phoneme (G2P) conversion

Module 3: Automatic Speech Recognition (ASR)
- Hidden Markov Models (HMM): Viterbi algorithm, Baum-Welch, forward-backward
- Gaussian Mixture Models (GMM): EM algorithm, model selection
- Deep Neural Network (DNN) acoustic models
- Connectionist Temporal Classification (CTC)
- Attention-based encoder-decoder: Transformer, LAS
- Beam search decoding, language model integration
- N-gram language models: interpolation, Kneser-Ney smoothing, perplexity
- Word Error Rate (WER), evaluation methodology
- Whisper, wav2vec 2.0, Conformer architectures
- Constrained decoding, logit biasing

Module 4: Speaker Recognition
- Speaker verification vs identification vs diarization
- i-vectors, x-vectors, TDNN architecture
- d-vectors, speaker embeddings, cosine similarity
- VoxCeleb dataset, speaker recognition evaluation
- Anti-spoofing, countermeasure systems
- LFCC, CQCC features for spoofing detection
- ASVspoof challenge, EER evaluation

Module 5: Text-to-Speech Synthesis (TTS)
- Statistical parametric speech synthesis
- WaveNet, Tacotron, FastSpeech, VITS architectures
- Neural vocoders: WaveRNN, HiFi-GAN, EnCodec
- Zero-shot voice cloning, speaker adaptation
- Prosody modeling: F0 generation, duration prediction
- Cross-lingual TTS, multilingual synthesis
- Meta MMS (Massively Multilingual Speech)

Module 6: Low-Resource Speech Processing
- Transfer learning, domain adaptation
- Data augmentation: SpecAugment, noise injection
- Self-supervised learning: wav2vec, HuBERT, data2vec
- Code-switching: Hinglish, multilingual ASR
- Language identification (LID), language boundary detection
- Low-resource languages: Maithili, Santhali, Gondi

Module 7: Robustness and Security
- Noise robustness, environmental mismatch
- Adversarial examples: FGSM, PGD attacks on speech models
- Adversarial perturbations, imperceptible noise
- Model interpretability, attention visualization
- Differential privacy in speech systems
- Voice anti-spoofing, presentation attack detection

Technical Terms Reference:
acoustic model, adaptation, adversarial, affricates, alignment,
allophone, amplitude, anti-spoofing, articulatory, attention,
autocorrelation, backpropagation, bandwidth, Baum-Welch,
beam search, cepstral, cepstrum, classification, coarticulation,
Conformer, connectionist, constrained decoding, corpus,
cosine similarity, countermeasure, CTC, d-vector, data augmentation,
decoding, deep learning, denoising, diarization, digital,
duration, embedding, encoder, energy, evaluation,
feature extraction, filterbank, formant, Fourier, frequency,
fundamental frequency, Gaussian, GMM, gradient, Griffin-Lim,
Hinglish, HMM, i-vector, interpolation, intonation, IPA,
Kneser-Ney, language identification, LFCC, LID, log-mel,
Maithili, Markov, MFCC, mixture model, multilingual,
neural network, noise, normalization, N-gram, perplexity,
phoneme, pitch, PLP, posterior, prosody, PYIN, RASTA,
recognition, resonance, rhythm, sampling, self-supervised,
signal, softmax, speaker, spectrogram, spectral subtraction,
SpecAugment, stochastic, stress, synthesis, TDNN,
Transformer, transfer learning, utterance, VAD, Viterbi,
vocoder, voice, vowel, wav2vec, WaveNet, WER, Whisper,
x-vector, YIN, zero-shot
"""


def convert_m4a_to_wav(m4a_path: str, output_dir: str) -> str:
    """Convert M4A to 16kHz mono WAV using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "original_segment.wav")

    # Check if already converted
    if os.path.exists(wav_path):
        logger.info(f"WAV already exists: {wav_path}")
        return wav_path

    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found.\n"
            "  macOS: brew install ffmpeg\n"
            "  macOS: brew install ffmpeg"
        )

    logger.info(f"Converting {m4a_path} → {wav_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", m4a_path,
        "-ar", "16000",         # Resample to 16kHz (ASR standard)
        "-ac", "1",             # Mono
        "-c:a", "pcm_s16le",    # 16-bit PCM
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

    size_mb = os.path.getsize(wav_path) / 1e6
    logger.info(f"Converted successfully: {wav_path} ({size_mb:.1f} MB)")
    return wav_path


def create_syllabus_file(output_dir: str) -> str:
    """Write course syllabus text for N-gram LM training."""
    syllabus_path = os.path.join(output_dir, "speech_syllabus.txt")
    with open(syllabus_path, "w", encoding="utf-8") as f:
        f.write(SPEECH_SYLLABUS)
    logger.info(f"Syllabus written: {syllabus_path}")
    return syllabus_path


def create_maithili_corpus(output_dir: str) -> str:
    """
    Export the bundled Maithili corpus to a JSON file for easy extension.
    Students can add more entries to this file.
    """
    from src.part2.lrl_translator import MAITHILI_CORPUS, MAITHILI_CONNECTORS
    corpus = {**MAITHILI_CORPUS, **MAITHILI_CONNECTORS}
    corpus_path = os.path.join(output_dir, "maithili_corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    logger.info(f"Maithili corpus exported: {corpus_path} ({len(corpus)} entries)")
    return corpus_path


def create_dummy_voice_ref(output_dir: str) -> str:
    """
    Create a placeholder 60s voice reference WAV (silence) if not provided.
    Replace with your actual voice recording!
    """
    ref_path = os.path.join(output_dir, "student_voice_ref.wav")
    if os.path.exists(ref_path):
        return ref_path

    import torch
    import torchaudio
    # 60 seconds of silence at 16kHz
    silence = torch.zeros(1, 16000 * 60)
    torchaudio.save(ref_path, silence, 16000)
    logger.warning(
        f"Created PLACEHOLDER voice reference: {ref_path}\n"
        "IMPORTANT: Replace with your actual 60-second voice recording!\n"
        "Record yourself reading for 60 seconds and save as student_voice_ref.wav"
    )
    return ref_path


def verify_setup(data_dir: str):
    """Verify all required data files exist."""
    required = [
        "original_segment.wav",
        "speech_syllabus.txt",
    ]
    optional = [
        "student_voice_ref.wav",
        "maithili_corpus.json",
    ]

    print("\n" + "=" * 50)
    print("DATA SETUP VERIFICATION")
    print("=" * 50)

    all_ok = True
    for f in required:
        path = os.path.join(data_dir, f)
        exists = os.path.exists(path)
        size = f"{os.path.getsize(path)/1e6:.1f}MB" if exists else "MISSING"
        status = "✓" if exists else "✗ REQUIRED"
        print(f"  {status} {f} ({size})")
        if not exists:
            all_ok = False

    for f in optional:
        path = os.path.join(data_dir, f)
        exists = os.path.exists(path)
        size = f"{os.path.getsize(path)/1e6:.1f}MB" if exists else "not found"
        status = "✓" if exists else "○ optional"
        print(f"  {status} {f} ({size})")

    print("=" * 50)
    if all_ok:
        print("✓ Setup complete! Run: python pipeline.py")
    else:
        print("✗ Setup incomplete. Fix missing files above.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Speech Assignment 2")
    parser.add_argument(
        "--input_m4a",
        type=str,
        default=None,
        help="Path to the provided M4A lecture recording",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Output data directory",
    )
    parser.add_argument(
        "--voice_ref",
        type=str,
        default=None,
        help="Path to your 60s voice reference (WAV). If not provided, creates placeholder.",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Convert M4A → WAV
    if args.input_m4a:
        convert_m4a_to_wav(args.input_m4a, args.data_dir)
    else:
        logger.warning(
            "No M4A file provided. Place your WAV file at data/original_segment.wav manually."
        )

    # Create supporting files
    create_syllabus_file(args.data_dir)
    create_maithili_corpus(args.data_dir)

    # Voice reference
    if args.voice_ref and os.path.exists(args.voice_ref):
        dst = os.path.join(args.data_dir, "student_voice_ref.wav")
        shutil.copy2(args.voice_ref, dst)
        logger.info(f"Voice reference copied: {dst}")
    else:
        create_dummy_voice_ref(args.data_dir)

    verify_setup(args.data_dir)
