"""
Task 3.1: Speaker Embedding Extraction (x-vector / d-vector).
Extracts high-dimensional speaker representation from 60s reference.
Architecture: TDNN-based x-vector extractor.
Author: Ashish Sinha (M25DE1047)
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.audio_utils import compute_mel_spectrogram, load_audio, TARGET_SR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# TDNN (Time-Delay Neural Network) x-vector architecture
# ─────────────────────────────────────────────────────────────────

class TDNNLayer(nn.Module):
    """
    Time-Delay Neural Network layer.
    Operates on a context window [−d, +d] around each frame.
    Implemented as Conv1d with dilation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=context_size,
            dilation=dilation,
            padding=(context_size - 1) * dilation // 2,
        )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


class StatisticsPooling(nn.Module):
    """
    Statistics Pooling: aggregates frame-level features into utterance-level.
    Concatenates mean and standard deviation over time dimension.
    This is the key operation that makes x-vectors utterance-level.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=-1)              # (B, C)
        std = x.std(dim=-1, unbiased=False)  # (B, C)
        return torch.cat([mean, std], dim=-1)  # (B, 2C)


class XVectorExtractorModel(nn.Module):
    """
    x-vector extractor based on Snyder et al. (2018).
    Architecture:
      5 TDNN layers (frame-level) → Statistics Pooling → 2 FC layers
    The embedding is taken from the first FC layer output (512-d).

    Design choice (non-obvious):
      The original x-vector paper uses 5 TDNN layers with specific
      context windows: {t-2..t+2}, {t-2..t+2}, {t-3..t+3}, {t}, {t}.
      This gives the network multi-scale temporal context (roughly
      40ms, 60ms, 90ms windows) to capture phoneme-level patterns
      for speaker characterization. We replicate this topology.
    """

    def __init__(self, input_dim: int = 80, embedding_dim: int = 512):
        super().__init__()

        # Frame-level TDNN layers (context windows match original paper)
        self.frame_layers = nn.Sequential(
            TDNNLayer(input_dim, 512, context_size=5, dilation=1),   # ±2 frames
            TDNNLayer(512, 512, context_size=5, dilation=2),          # ±4 frames
            TDNNLayer(512, 512, context_size=7, dilation=3),          # ±9 frames
            TDNNLayer(512, 512, context_size=1, dilation=1),
            TDNNLayer(512, 1500, context_size=1, dilation=1),
        )
        # Use track_running_stats=False to handle single-sample inference
        for m in self.frame_layers.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = True

        # Statistics pooling
        self.stats_pooling = StatisticsPooling()  # output: 3000

        # Segment-level (utterance) layers
        self.segment1 = nn.Linear(3000, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.segment2 = nn.Linear(embedding_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # Speaker classification head (used during training only)
        self.classifier = nn.Linear(embedding_dim, 1000)  # 1000 speakers

    def forward(
        self, x: torch.Tensor, return_embedding: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, T) mel features
        Returns:
            embedding: (B, embedding_dim) if return_embedding=True
            logits:    (B, n_speakers) otherwise
        """
        # Frame-level processing
        feat = self.frame_layers(x)       # (B, 1500, T)

        # Statistics pooling → utterance level
        pooled = self.stats_pooling(feat)  # (B, 3000)

        # Embedding layers
        emb = F.relu(self.bn1(self.segment1(pooled)))  # (B, 512)
        emb = F.relu(self.bn2(self.segment2(emb)))     # (B, 512) ← x-vector

        if return_embedding:
            return emb

        return self.classifier(emb)


class VoiceEmbeddingExtractor:
    """
    Wrapper for extracting speaker embeddings from audio.
    Handles loading, chunking, and averaging for long references.
    """

    def __init__(
        self,
        model_type: str = "xvector",
        embedding_dim: int = 512,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
    ):
        self.device = device
        self.embedding_dim = embedding_dim

        self.model = XVectorExtractorModel(
            input_dim=80,
            embedding_dim=embedding_dim,
        ).to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(ckpt)
            logger.info(f"Loaded x-vector checkpoint: {checkpoint_path}")
        else:
            logger.warning(
                "No x-vector checkpoint found. Embedding will be random (needs training). "
                "For production use, fine-tune on VoxCeleb2 or NIST SRE data."
            )

        self.model.eval()

    def extract(
        self,
        audio_path: str,
        chunk_duration_sec: float = 3.0,
        output_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Extract speaker embedding from reference audio.
        For robust estimation: chunk the audio, extract per-chunk,
        then average (mean-pool) the embeddings.

        Args:
            audio_path: path to 60s reference audio
            chunk_duration_sec: chunk size for chunked extraction
        Returns:
            embedding: (embedding_dim,) tensor
        """
        waveform, sr = load_audio(audio_path, target_sr=TARGET_SR)
        duration = waveform.shape[-1] / sr
        logger.info(f"Extracting x-vector from {duration:.1f}s reference audio")

        chunk_samples = int(chunk_duration_sec * sr)
        hop_length = 160
        embeddings = []

        for start in range(0, waveform.shape[-1] - chunk_samples, chunk_samples // 2):
            chunk = waveform[:, start: start + chunk_samples]
            mel = compute_mel_spectrogram(chunk, sample_rate=sr)  # (80, T)
            mel_tensor = mel.unsqueeze(0).to(self.device)          # (1, 80, T)

            with torch.no_grad():
                emb = self.model(mel_tensor, return_embedding=True)  # (1, 512)
            embeddings.append(emb.squeeze(0))

        if not embeddings:
            # Short audio fallback
            mel = compute_mel_spectrogram(waveform, sample_rate=sr)
            mel_tensor = mel.unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model(mel_tensor, return_embedding=True)
            embeddings = [emb.squeeze(0)]

        # Mean pooling across chunks = robust utterance-level embedding
        speaker_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        # L2 normalize
        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)

        logger.info(
            f"Speaker embedding shape: {speaker_embedding.shape}, "
            f"norm: {speaker_embedding.norm().item():.4f}"
        )

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            torch.save(speaker_embedding.cpu(), output_path)
            logger.info(f"Embedding saved: {output_path}")

        return speaker_embedding
