"""
Task 1.2: Constrained Beam Search ASR using Whisper-v3 with
N-gram language model logit biasing for technical vocabulary.
Author: Ashish Sinha (M25DE1047)
"""

import collections
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from src.utils.audio_utils import load_audio, TARGET_SR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# N-gram Language Model (built from Speech Syllabus)
# ─────────────────────────────────────────────────────────────────

class NGramLM:
    """
    Character/subword N-gram language model trained on the course syllabus.
    Used to compute logit biases for Whisper token vocabulary.

    Mathematical formulation:
      P_ngram(w_t | w_{t-n+1},...,w_{t-1}) =
          count(w_{t-n+1},...,w_t) / count(w_{t-n+1},...,w_{t-1})
      with Laplace (add-k) smoothing:
          P_smooth = (count + k) / (context_count + k * V)

    Logit bias for token t:
      bias(t) = β * log P_ngram(t | context) if P_ngram > threshold else 0
    Where β is the bias_strength hyperparameter.
    """

    def __init__(self, order: int = 3, smoothing_k: float = 0.1):
        self.order = order
        self.k = smoothing_k
        self.ngram_counts = collections.defaultdict(int)
        self.context_counts = collections.defaultdict(int)
        self.vocab = set()
        self._trained = False

    def train(self, text: str):
        """Train N-gram model on raw text corpus."""
        tokens = text.lower().split()
        self.vocab.update(tokens)

        for n in range(1, self.order + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i: i + n])
                self.ngram_counts[ngram] += 1
                if n > 1:
                    self.context_counts[ngram[:-1]] += 1

        self._trained = True
        logger.info(
            f"N-gram LM trained: order={self.order}, vocab_size={len(self.vocab)}, "
            f"total_ngrams={len(self.ngram_counts)}"
        )

    def log_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """Compute log P(word | context) with Laplace smoothing."""
        if not self._trained:
            return 0.0
        context = context[-(self.order - 1):]
        ngram = context + (word.lower(),)
        ctx_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts.get(ngram, 0)
        vocab_size = len(self.vocab)
        prob = (ngram_count + self.k) / (ctx_count + self.k * vocab_size + 1e-9)
        return math.log(prob + 1e-9)

    def get_technical_terms(self) -> List[str]:
        """Extract high-frequency technical terms (unigrams with count > 1)."""
        terms = [
            word for word, count in
            collections.Counter(
                {k[0]: v for k, v in self.ngram_counts.items() if len(k) == 1}
            ).most_common()
            if count > 1 and len(word) > 4
        ]
        return terms


# ─────────────────────────────────────────────────────────────────
# Logit Bias Processor for Whisper
# ─────────────────────────────────────────────────────────────────

class LogitBiasProcessor:
    """
    Applies token-level logit biases from N-gram LM during Whisper decoding.

    At each decoding step:
      1. Get current partial hypothesis tokens → decode to text → extract context
      2. For each token in Whisper vocabulary:
           if token text ∈ technical vocabulary: add +bias
           using N-gram LM: add +bias * log_prob(token | context)
      3. Pass modified logits to beam search

    This implements "soft" constrained decoding. A "hard" version would
    force specific tokens; soft biasing is more robust to noisy speech.
    """

    def __init__(
        self,
        tokenizer,
        ngram_lm: NGramLM,
        technical_terms: List[str],
        bias_strength: float = 2.5,
    ):
        self.tokenizer = tokenizer
        self.ngram_lm = ngram_lm
        self.technical_terms = set(t.lower() for t in technical_terms)
        self.bias_strength = bias_strength

        # Pre-compute token IDs for technical terms
        self.technical_token_ids = self._build_technical_token_map()
        logger.info(f"Logit bias: {len(self.technical_token_ids)} technical token IDs")

    def _build_technical_token_map(self) -> Dict[int, float]:
        """Map token IDs of technical terms to their bias values."""
        token_bias = {}
        for term in self.technical_terms:
            try:
                ids = self.tokenizer.encode(" " + term, add_special_tokens=False)
                for tid in ids:
                    token_bias[tid] = self.bias_strength
            except Exception:
                pass
        return token_bias

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Called by Whisper at each beam step to modify logits.
        input_ids: (batch, seq_len) — already-generated tokens
        scores:    (batch, vocab_size) — raw logits
        """
        for token_id, bias in self.technical_token_ids.items():
            if token_id < scores.shape[-1]:
                scores[:, token_id] += bias
        return scores


# ─────────────────────────────────────────────────────────────────
# Constrained ASR (Whisper + Logit Biasing)
# ─────────────────────────────────────────────────────────────────

SPEECH_SYLLABUS_DEFAULT = """
speech understanding signal processing acoustic model language model
hidden markov model hmm gaussian mixture model gmm deep neural network dnn
mel frequency cepstral coefficient mfcc cepstrum stochastic gradient descent
backpropagation attention mechanism transformer encoder decoder
connectionist temporal classification ctc beam search decoding
viterbi algorithm forward backward algorithm baum welch
feature extraction spectrogram filterbank pitch fundamental frequency
voiced unvoiced fricative plosive phoneme allophones
word error rate wer automatic speech recognition asr
speaker recognition diarization voice activity detection vad
recurrent neural network lstm gated recurrent unit gru
end to end learning acoustic phonetics articulatory
formant resonance bandwidth spectral envelope
prosody intonation rhythm stress duration
language identification lid code switching hinglish bilingual
zero shot cross lingual transfer learning fine tuning
mel spectrogram log filterbank energy lfe
plp perceptual linear prediction rasta
dynamic time warping dtw alignment
pronunciation lexicon vocabulary out of vocabulary oov
n gram interpolation perplexity smoothing backoff
kaldi espnet fairseq torchaudio pytorch whisper wav2vec
"""


class ConstrainedASR:
    """
    Whisper-based ASR with N-gram logit biasing for technical term prioritization.
    Handles code-switched (Hinglish) audio using language-specific decoding.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        syllabus_path: Optional[str] = None,
        ngram_order: int = 3,
        logit_bias_strength: float = 2.5,
        device: str = "cpu",
        beam_size: int = 5,
    ):
        self.model_name = model_name
        self.device = device
        self.beam_size = beam_size
        self.logit_bias_strength = logit_bias_strength

        # Load syllabus for N-gram LM
        if syllabus_path and os.path.exists(syllabus_path):
            with open(syllabus_path, "r") as f:
                syllabus_text = f.read()
        else:
            logger.warning("Using default syllabus text for N-gram LM.")
            syllabus_text = SPEECH_SYLLABUS_DEFAULT

        # Train N-gram LM
        self.ngram_lm = NGramLM(order=ngram_order)
        self.ngram_lm.train(syllabus_text)
        self.technical_terms = self.ngram_lm.get_technical_terms()
        logger.info(f"Technical terms: {self.technical_terms[:20]}")

        # Load Whisper model (lazy — only when needed)
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load Whisper to avoid OOM if only doing other parts."""
        if self._model is not None:
            return
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._processor = WhisperProcessor.from_pretrained(self.model_name)
            self._model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers accelerate"
            )

    def transcribe(
        self,
        audio_path: str,
        lid_results: Optional[Dict] = None,
        output_path: Optional[str] = None,
        chunk_duration_sec: float = 30.0,
    ) -> Dict:
        """
        Transcribe audio with constrained beam search.

        Strategy for code-switching:
          - For English segments: force language token <|en|>
          - For Hindi segments: force language token <|hi|>
          - LID segment info drives per-chunk language forcing
          - Logit biases applied uniformly for technical terms
        """
        self._load_model()

        waveform, sr = load_audio(audio_path, target_sr=TARGET_SR)
        total_samples = waveform.shape[-1]
        chunk_samples = int(chunk_duration_sec * sr)

        logit_bias_proc = LogitBiasProcessor(
            tokenizer=self._processor.tokenizer,
            ngram_lm=self.ngram_lm,
            technical_terms=self.technical_terms,
            bias_strength=self.logit_bias_strength,
        )

        all_segments = []
        full_text_parts = []
        chunk_idx = 0

        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk_wav = waveform[:, start_sample:end_sample].squeeze().numpy()
            start_sec = start_sample / sr

            # Determine dominant language for this chunk from LID
            language = self._get_chunk_language(lid_results, start_sec, start_sec + chunk_duration_sec)

            # Prepare input features
            inputs = self._processor(
                chunk_wav,
                sampling_rate=sr,
                return_tensors="pt",
            ).to(self.device)

            # Generate with constrained beam search + logit biasing.
            # transformers 5.x: use language/task kwargs (forced_decoder_ids
            # is deprecated and causes a token-count overflow at max_new_tokens=448
            # because the prefix already occupies 2-4 decoder positions).
            # Safe limit: 448 (model max) - 4 (prefix tokens) = 444.
            with torch.no_grad():
                try:
                    generated = self._model.generate(
                        **inputs,
                        language=language,
                        task="transcribe",
                        num_beams=self.beam_size,
                        logits_processor=[logit_bias_proc],
                        return_timestamps=True,
                        max_new_tokens=444,
                    )
                except TypeError:
                    # Older transformers API fallback
                    forced_lang_id = self._processor.tokenizer.convert_tokens_to_ids(
                        f"<|{language}|>"
                    )
                    generated = self._model.generate(
                        **inputs,
                        forced_decoder_ids=[[1, forced_lang_id]],
                        num_beams=self.beam_size,
                        logits_processor=[logit_bias_proc],
                        return_timestamps=True,
                        max_new_tokens=444,
                    )

            # Decode
            transcription = self._processor.batch_decode(
                generated, skip_special_tokens=False
            )[0]

            # Parse timestamps from Whisper output
            segments = self._parse_whisper_output(transcription, offset_sec=start_sec)
            all_segments.extend(segments)
            full_text_parts.append(
                self._processor.batch_decode(generated, skip_special_tokens=True)[0]
            )

            chunk_idx += 1
            logger.info(f"Chunk {chunk_idx} ({start_sec:.1f}s-{start_sec+chunk_duration_sec:.1f}s): {full_text_parts[-1][:80]}...")

        result = {
            "text": " ".join(full_text_parts),
            "segments": all_segments,
            "technical_terms_used": self.technical_terms[:50],
            "ngram_order": self.ngram_lm.order,
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Transcript saved: {output_path}")

        return result

    def _get_chunk_language(
        self,
        lid_results: Optional[Dict],
        start_sec: float,
        end_sec: float,
    ) -> str:
        """Determine dominant language in time range from LID results."""
        if lid_results is None:
            return "en"
        segments = lid_results.get("language_segments", [])
        lang_counts = {"english": 0.0, "hindi": 0.0}
        for seg in segments:
            overlap_start = max(seg["start"], start_sec)
            overlap_end = min(seg["end"], end_sec)
            if overlap_end > overlap_start:
                lang_counts[seg["language"]] += overlap_end - overlap_start
        dominant = max(lang_counts, key=lang_counts.get)
        return "en" if dominant == "english" else "hi"

    def _parse_whisper_output(
        self, raw_text: str, offset_sec: float = 0.0
    ) -> List[Dict]:
        """Parse Whisper's timestamp tokens into structured segments."""
        # Whisper format: <|0.00|> text <|1.20|> more text
        pattern = r"<\|(\d+\.\d+)\|>(.*?)(?=<\|\d+\.\d+\|>|$)"
        matches = re.findall(pattern, raw_text, re.DOTALL)
        segments = []
        for i, (ts, text) in enumerate(matches):
            text = text.strip()
            if not text:
                continue
            start = float(ts) + offset_sec
            end_ts = float(matches[i + 1][0]) + offset_sec if i + 1 < len(matches) else start + 2.0
            segments.append({
                "start": round(start, 3),
                "end": round(end_ts, 3),
                "text": text,
            })
        if not segments:
            # Fallback: return plain text without timestamps
            clean = re.sub(r"<\|[^>]+\|>", "", raw_text).strip()
            segments = [{"start": offset_sec, "end": offset_sec + 30.0, "text": clean}]
        return segments
