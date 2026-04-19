"""
Task 3.2: Prosody Feature Extraction & Dynamic Time Warping (DTW).
Extracts F0 (pitch) and energy from source lecture, warps to target.
Author: Ashish Sinha (M25DE1047)
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

from src.utils.audio_utils import (
    compute_energy_contour,
    frames_to_time,
    load_audio,
    TARGET_SR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# F0 (Fundamental Frequency) Extraction — PYIN algorithm
# ─────────────────────────────────────────────────────────────────

class F0Extractor:
    """
    PYIN (Probabilistic YIN) F0 estimation.
    PYIN improves on YIN by computing probability distribution over
    F0 candidates, giving better results in noise and at voicing boundaries.

    Implementation uses torchaudio's built-in if available,
    otherwise falls back to librosa's PYIN or a manual YIN implementation.
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SR,
        frame_length: int = 512,
        hop_length: int = 160,
        f0_min: float = 75.0,     # Hz — minimum human F0 (bass voice)
        f0_max: float = 600.0,    # Hz — maximum (high-pitched female)
    ):
        self.sr = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract(self, waveform: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 contour and voiced/unvoiced mask.

        Returns:
            f0: (T,) array of F0 in Hz (0 for unvoiced frames)
            voiced: (T,) boolean array
        """
        wav_np = waveform.squeeze().numpy()

        # Try librosa PYIN (most accurate)
        try:
            import librosa
            f0, voiced_flag, voiced_prob = librosa.pyin(
                wav_np,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length,
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0, voiced_flag
        except ImportError:
            logger.warning("librosa not available. Using autocorrelation-based F0.")
            return self._autocorr_f0(wav_np)

    def _autocorr_f0(self, wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified autocorrelation F0 tracker.
        Less accurate than PYIN but has no external dependencies.
        """
        num_frames = (len(wav) - self.frame_length) // self.hop_length + 1
        f0 = np.zeros(num_frames)
        voiced = np.zeros(num_frames, dtype=bool)

        min_period = int(self.sr / self.f0_max)
        max_period = int(self.sr / self.f0_min)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = wav[start: start + self.frame_length]
            if len(frame) < self.frame_length:
                continue

            # Autocorrelation
            acorr = np.correlate(frame, frame, mode="full")
            acorr = acorr[len(acorr) // 2:]

            # Find peak in valid period range
            search = acorr[min_period: max_period + 1]
            if len(search) == 0:
                continue
            peak_idx = np.argmax(search) + min_period
            peak_val = acorr[peak_idx] / (acorr[0] + 1e-9)

            if peak_val > 0.3:  # Voiced threshold
                f0[i] = self.sr / peak_idx
                voiced[i] = True

        return f0, voiced


# ─────────────────────────────────────────────────────────────────
# DTW Prosody Warping
# ─────────────────────────────────────────────────────────────────

class ProsodyWarper:
    """
    DTW-based prosody warping for TTS.

    Extracts F0 + energy from source (professor's lecture),
    then applies DTW alignment to map prosodic shape onto
    synthesized LRL speech, preserving "teaching style."

    Mathematical formulation of DTW warping:
      Given source prosody P_s of length N and reference P_r of length M,
      DTW finds alignment path π* = argmin_{π} Σ d(P_s[π_n], P_r[π_m])
      Subject to: boundary (start/end), monotonicity, step constraints.
      The warped prosody P_w is P_s resampled along the optimal path.

    Design choice (non-obvious):
      We warp F0 in the log domain (log-F0) rather than linear Hz,
      because perceptual pitch distance is logarithmic (musical intervals).
      A ±100Hz error at 200Hz base sounds the same as ±200Hz at 400Hz.
      DTW on log-F0 gives perceptually better prosody transfer.
    """

    def __init__(
        self,
        f0_extractor: str = "pyin",
        energy_frame_size: int = 512,
        dtw_metric: str = "euclidean",
        device: str = "cpu",
        hop_length: int = 160,
        sample_rate: int = TARGET_SR,
    ):
        self.hop_length = hop_length
        self.sr = sample_rate
        self.energy_frame_size = energy_frame_size
        self.f0_extractor_obj = F0Extractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

    def warp(
        self,
        source_path: str,
        target_length_frames: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Extract prosody from source audio and compute warping template.

        Returns:
            dict with f0_contour, energy_contour, dtw_path, statistics
        """
        waveform, sr = load_audio(source_path, target_sr=self.sr)
        logger.info(f"Extracting prosody from {source_path}")

        # Extract F0 and energy
        f0, voiced = self.f0_extractor_obj.extract(waveform)
        energy = compute_energy_contour(
            waveform,
            frame_size=self.energy_frame_size,
            hop_size=self.hop_length,
        )

        # Align lengths
        min_len = min(len(f0), len(energy))
        f0 = f0[:min_len]
        voiced = voiced[:min_len]
        energy = energy[:min_len]

        # Log-F0 (voiced frames only, 0 elsewhere)
        log_f0 = np.zeros_like(f0)
        log_f0[voiced] = np.log(f0[voiced] + 1e-9)

        # Normalize energy
        energy_norm = (energy - energy.mean()) / (energy.std() + 1e-9)

        # Build combined prosody feature: [log_f0, energy]
        prosody_feature = np.stack([log_f0, energy_norm], axis=1)  # (T, 2)

        # Statistics for conditioning TTS
        stats = {
            "f0_mean_hz": float(f0[voiced].mean()) if voiced.any() else 120.0,
            "f0_std_hz": float(f0[voiced].std()) if voiced.any() else 20.0,
            "f0_min_hz": float(f0[voiced].min()) if voiced.any() else 80.0,
            "f0_max_hz": float(f0[voiced].max()) if voiced.any() else 300.0,
            "voiced_ratio": float(voiced.mean()),
            "energy_mean": float(energy.mean()),
            "energy_std": float(energy.std()),
            "duration_frames": int(min_len),
            "duration_seconds": round(min_len * self.hop_length / self.sr, 3),
        }

        logger.info(
            f"Prosody: F0 mean={stats['f0_mean_hz']:.1f}Hz, "
            f"voiced={stats['voiced_ratio']:.1%}, "
            f"duration={stats['duration_seconds']:.1f}s"
        )

        result = {
            "f0": f0.tolist(),
            "log_f0": log_f0.tolist(),
            "energy": energy.tolist(),
            "voiced": voiced.tolist(),
            "prosody_feature": prosody_feature.tolist(),
            "statistics": stats,
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            torch.save(result, output_path)
            logger.info(f"Prosody features saved: {output_path}")

        return result

    def apply_dtw_warping(
        self,
        source_prosody: np.ndarray,
        target_prosody: np.ndarray,
    ) -> np.ndarray:
        """
        Apply DTW to align source prosody onto target length.

        Args:
            source_prosody: (T_src, D) — professor's prosody features
            target_prosody: (T_tgt, D) — initial synthesized prosody
        Returns:
            warped_prosody: (T_tgt, D) — source prosody warped to target length
        """
        from scipy.spatial.distance import cdist
        from scipy.interpolate import interp1d

        # DTW cost matrix (on log-F0 only for stability)
        cost = cdist(source_prosody[:, 0:1], target_prosody[:, 0:1], metric="euclidean")

        # DP
        n, m = cost.shape
        dp = np.full((n, m), np.inf)
        dp[0, 0] = cost[0, 0]
        for i in range(1, n):
            dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
        for j in range(1, m):
            dp[0, j] = dp[0, j - 1] + cost[0, j]
        for i in range(1, n):
            for j in range(1, m):
                dp[i, j] = cost[i, j] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

        # Traceback
        path = []
        i, j = n - 1, m - 1
        while i > 0 or j > 0:
            path.append((i, j))
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                step = np.argmin([dp[i-1, j], dp[i, j-1], dp[i-1, j-1]])
                if step == 0: i -= 1
                elif step == 1: j -= 1
                else: i -= 1; j -= 1
        path.append((0, 0))
        path = list(reversed(path))

        # Resample source prosody to target length using DTW path
        src_indices = np.array([p[0] for p in path])
        tgt_indices = np.array([p[1] for p in path])

        warped = np.zeros((m, source_prosody.shape[1]))
        for dim in range(source_prosody.shape[1]):
            interp = interp1d(
                tgt_indices, source_prosody[src_indices, dim],
                kind="linear", fill_value="extrapolate",
            )
            warped[:, dim] = interp(np.arange(m))

        return warped


def Optional(*args):
    """Type hint shim"""
    pass

from typing import Optional  # noqa: F811
