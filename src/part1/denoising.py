"""
Task 1.3: Denoising & Normalization for classroom audio.
Implements Spectral Subtraction (no extra deps) and optional DeepFilterNet.

Author: Rohit (M25DE1047), IIT Jodhpur
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
import torchaudio.transforms as T

from src.utils.audio_utils import load_audio, save_audio, TARGET_SR

logger = logging.getLogger(__name__)


def _deepfilternet_available() -> bool:
    """Check at runtime whether deepfilternet is installed."""
    try:
        import df  # noqa: F401
        return True
    except ImportError:
        return False


class AudioDenoiser:
    """
    Classroom speech denoiser with two backends:

    1. ``spectral_subtraction`` (default, zero extra dependencies)
       Power-spectrum spectral subtraction with over-subtraction (α) and
       spectral floor (β) to prevent musical noise artifacts.

    2. ``deepfilternet`` (optional, requires: pip install deepfilternet)
       Neural RNN denoiser — higher quality but needs Rust on macOS.
       On macOS Apple Silicon install with: brew install rust && pip install deepfilternet
       Falls back to spectral_subtraction automatically if not available.

    Design choice (non-obvious):
      Spectral subtraction estimates noise PSD from the first ~200ms of audio
      (assumed to be background noise before speech starts), then subtracts a
      scaled version from all frames. Half-wave rectification
      (max(result, β·|Y|²)) prevents "musical noise" — the tonal artifacts
      that naive subtraction produces at frequencies where the noise PSD
      estimate exceeds the true noise. α=2.0 and β=0.002 are tuned for
      typical Indian classroom reverb (RT60 ≈ 0.4-0.8s).
    """

    def __init__(
        self,
        method: str = "spectral_subtraction",
        noise_estimation_frames: int = 20,
        over_subtraction_alpha: float = 2.0,
        spectral_floor_beta: float = 0.002,
        device: str = "cpu",
    ):
        self.device = device
        self.noise_frames = noise_estimation_frames
        self.alpha = over_subtraction_alpha
        self.beta = spectral_floor_beta

        # Resolve method: fall back to spectral_subtraction if deepfilternet
        # is requested but not installed (common on macOS without Rust)
        if method == "deepfilternet" and not _deepfilternet_available():
            logger.warning(
                "deepfilternet not installed — falling back to spectral_subtraction.\n"
                "  macOS install: brew install rust && pip install deepfilternet\n"
                "  Linux install: pip install deepfilternet"
            )
            method = "spectral_subtraction"
        self.method = method

    # ── public API ────────────────────────────────────────────────

    def process(self, input_path: str, output_path: str) -> str:
        """Load audio, apply chosen denoiser, save result."""
        waveform, sr = load_audio(input_path, target_sr=TARGET_SR)
        logger.info(
            f"Denoising [{self.method}]: {input_path} "
            f"({waveform.shape[-1]/sr:.1f}s @ {sr}Hz)"
        )

        if self.method == "spectral_subtraction":
            clean_np = self._spectral_subtraction(waveform.squeeze().numpy(), sr)
            clean = torch.from_numpy(clean_np).unsqueeze(0)
        elif self.method == "deepfilternet":
            clean = self._deepfilternet(waveform, sr)
        else:
            raise ValueError(f"Unknown denoising method: {self.method!r}")

        save_audio(clean, output_path, sample_rate=sr)
        logger.info(f"Saved denoised audio: {output_path}")
        return output_path

    # ── spectral subtraction ──────────────────────────────────────

    def _spectral_subtraction(
        self,
        wav: np.ndarray,
        sr: int,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
    ) -> np.ndarray:
        """
        Power-spectrum spectral subtraction.

        Algorithm:
          1. STFT with Hanning window
          2. Estimate noise PSD from first `noise_frames` frames
          3. Subtract α × noise PSD from signal PSD
          4. Apply spectral floor β × |Y(f)|²
          5. Reconstruct signal via ISTFT (original phase)
          6. Overlap-add with window normalization
          7. Peak-normalize output
        """
        window = np.hanning(win_length)
        wav_padded = np.pad(wav, win_length // 2)
        num_frames = (len(wav_padded) - win_length) // hop_length + 1

        # STFT
        stft = np.zeros((n_fft // 2 + 1, num_frames), dtype=complex)
        for i in range(num_frames):
            start = i * hop_length
            frame = wav_padded[start: start + win_length]
            if len(frame) < win_length:
                frame = np.pad(frame, (0, win_length - len(frame)))
            padded = np.zeros(n_fft)
            padded[:win_length] = frame * window
            stft[:, i] = np.fft.rfft(padded)

        # Noise PSD from initial frames
        noise_psd = np.mean(
            np.abs(stft[:, : self.noise_frames]) ** 2, axis=1, keepdims=True
        )

        # Spectral subtraction with floor
        signal_psd = np.abs(stft) ** 2
        clean_psd = np.maximum(
            signal_psd - self.alpha * noise_psd,
            self.beta * signal_psd,
        )

        # Reconstruct — keep original phase
        clean_stft = np.sqrt(clean_psd) * np.exp(1j * np.angle(stft))

        # ISTFT via overlap-add
        out = np.zeros(len(wav_padded))
        win_sum = np.zeros(len(wav_padded))
        for i in range(num_frames):
            frame_t = np.fft.irfft(clean_stft[:, i], n=n_fft)[:win_length] * window
            s = i * hop_length
            out[s: s + win_length] += frame_t
            win_sum[s: s + win_length] += window ** 2

        nz = win_sum > 1e-8
        out[nz] /= win_sum[nz]

        # Trim padding and peak-normalise
        out = out[win_length // 2: win_length // 2 + len(wav)]
        peak = np.abs(out).max()
        if peak > 0:
            out /= peak
        return out.astype(np.float32)

    # ── deepfilternet (optional) ──────────────────────────────────

    def _deepfilternet(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        DeepFilterNet2 neural denoiser.
        Only called when deepfilternet is confirmed available at __init__.
        DeepFilterNet operates at 48kHz; we resample in/out.
        """
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df()

        # DeepFilterNet expects 48kHz mono
        resamp_up   = T.Resample(sr, 48000)
        resamp_down = T.Resample(48000, sr)

        wf_48k = resamp_up(waveform)
        enhanced_48k = enhance(model, df_state, wf_48k)
        return resamp_down(enhanced_48k)
