"""
Audio utility functions shared across all pipeline stages.
Author: Ashish Sinha (M25DE1047)
"""

import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

TARGET_SR = 16000   # Standard for ASR models


def load_audio(
    path: str,
    target_sr: int = TARGET_SR,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file, resample to target_sr, optionally convert to mono.

    Returns:
        waveform: (1, T) float32 tensor
        sample_rate: int
    """
    waveform, sr = torchaudio.load(path)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    if normalize:
        # Peak normalization to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    return waveform, sr


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = TARGET_SR):
    """Save waveform tensor to WAV file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform.cpu().float(), sample_rate)
    logger.debug(f"Saved audio: {path} ({waveform.shape[-1] / sample_rate:.2f}s)")


def get_audio_segment(
    input_path: str,
    output_path: str,
    start_sec: float = 0.0,
    duration_sec: float = 600.0,
    target_sr: int = TARGET_SR,
):
    """
    Extract a fixed-duration segment from audio.
    Converts M4A → WAV and resamples if needed.
    """
    waveform, sr = load_audio(input_path, target_sr=target_sr)
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    end_sample = min(end_sample, waveform.shape[-1])

    segment = waveform[:, start_sample:end_sample]
    save_audio(segment, output_path, sample_rate=sr)
    actual_dur = segment.shape[-1] / sr
    logger.info(f"Extracted segment: {actual_dur:.2f}s → {output_path}")
    return output_path


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SR,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.Tensor:
    """
    Compute log-mel spectrogram features.
    Returns: (n_mels, T) tensor
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0,
        power=2.0,
        normalized=False,
    )
    mel = mel_transform(waveform)
    log_mel = torch.log(mel + 1e-9)
    return log_mel.squeeze(0)  # (n_mels, T)


def compute_lfcc(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SR,
    n_lfcc: int = 60,
    n_fft: int = 512,
    hop_length: int = 160,
) -> torch.Tensor:
    """
    Compute LFCC (Linear Frequency Cepstral Coefficients).
    Used for anti-spoofing countermeasure.
    Unlike MFCC (mel-scale filterbank), LFCC uses linearly-spaced filters.
    This makes it sensitive to fine spectral details important for detecting
    synthesized speech artifacts.
    """
    # Linear filterbank: evenly spaced in Hz (not mel)
    lfcc_transform = T.LFCC(
        sample_rate=sample_rate,
        n_filter=128,
        f_min=0.0,
        f_max=sample_rate // 2,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "center": False,
        },
    )
    lfcc = lfcc_transform(waveform)
    return lfcc.squeeze(0)  # (n_lfcc, T)


def compute_energy_contour(
    waveform: torch.Tensor,
    frame_size: int = 512,
    hop_size: int = 160,
) -> np.ndarray:
    """
    Compute per-frame RMS energy (short-time energy).
    Used for prosody analysis.
    """
    wav_np = waveform.squeeze().numpy()
    num_frames = (len(wav_np) - frame_size) // hop_size + 1
    energy = np.array([
        np.sqrt(np.mean(wav_np[i * hop_size: i * hop_size + frame_size] ** 2))
        for i in range(num_frames)
    ])
    return energy


def snr_db(original: torch.Tensor, noisy: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    noise = noisy - original
    signal_power = (original ** 2).mean().item()
    noise_power = (noise ** 2).mean().item()
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / (noise_power + 1e-12))


def frames_to_time(frame_idx: int, hop_length: int = 160, sr: int = TARGET_SR) -> float:
    """Convert frame index to time in seconds."""
    return (frame_idx * hop_length) / sr


def time_to_frames(time_sec: float, hop_length: int = 160, sr: int = TARGET_SR) -> int:
    """Convert time in seconds to frame index."""
    return int(time_sec * sr / hop_length)
