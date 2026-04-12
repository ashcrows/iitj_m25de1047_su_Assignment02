"""
Quick sanity-check test: verifies all modules import and basic operations work
without requiring any pretrained models or the actual audio file.

Usage:
    python tests/test_pipeline.py
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING)

import torch
import numpy as np

PASS = "✓"
FAIL = "✗"
results = []


def test(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append(False)


print("\n" + "=" * 55)
print("Speech Assignment 2 — Module Sanity Tests")
print("=" * 55)

# ── Audio utils ────────────────────────────────────────────
print("\n[Utils]")

def test_mel():
    from src.utils.audio_utils import compute_mel_spectrogram
    wav = torch.randn(1, 16000)
    mel = compute_mel_spectrogram(wav, sample_rate=16000)
    assert mel.shape[0] == 80, f"Expected 80 mel bins, got {mel.shape[0]}"
test("compute_mel_spectrogram", test_mel)

def test_lfcc():
    from src.utils.audio_utils import compute_lfcc
    wav = torch.randn(1, 16000)
    lfcc = compute_lfcc(wav, sample_rate=16000, n_lfcc=60)
    assert lfcc.shape[0] == 60
test("compute_lfcc", test_lfcc)

def test_snr():
    from src.utils.audio_utils import snr_db
    x = torch.randn(1, 16000)
    x_noisy = x + 0.001 * torch.randn_like(x)
    s = snr_db(x, x_noisy)
    assert s > 30, f"SNR should be high for small noise, got {s:.1f}"
test("snr_db", test_snr)

def test_energy():
    from src.utils.audio_utils import compute_energy_contour
    wav = torch.randn(1, 16000)
    energy = compute_energy_contour(wav)
    assert len(energy) > 0
test("compute_energy_contour", test_energy)

# ── Evaluation metrics ─────────────────────────────────────
print("\n[Evaluation]")

def test_wer():
    from src.utils.evaluation import compute_wer
    wer = compute_wer("the cat sat on the mat", "the cat sat on the mat")
    assert wer == 0.0
    wer2 = compute_wer("hello world", "hello earth universe")
    assert 0.0 < wer2 <= 1.0
test("compute_wer (exact match + mismatch)", test_wer)

def test_eer():
    from src.utils.evaluation import compute_eer
    # Perfect separation
    bf = np.array([0.9, 0.8, 0.85])
    sp = np.array([0.1, 0.2, 0.15])
    eer = compute_eer(bf, sp)
    assert eer < 0.1, f"EER should be low for perfect separation: {eer}"
    # Random (EER ≈ 0.5)
    bf_rand = np.random.rand(100)
    sp_rand = np.random.rand(100)
    eer_rand = compute_eer(bf_rand, sp_rand)
    assert 0.3 < eer_rand < 0.7
test("compute_eer", test_eer)

def test_confusion():
    from src.utils.evaluation import compute_lid_confusion_matrix
    pred = [0, 0, 1, 1, 0, 1]
    ref  = [0, 1, 1, 1, 0, 0]
    cm = compute_lid_confusion_matrix(pred, ref)
    assert "matrix" in cm
    assert "f1_scores" in cm
    assert "macro_f1" in cm
test("compute_lid_confusion_matrix", test_confusion)

def test_mcd():
    from src.utils.evaluation import compute_mcd
    ref = np.random.randn(100, 13)
    syn = np.random.randn(80, 13)
    mcd = compute_mcd(ref, syn)
    assert 0 < mcd < 100
test("compute_mcd (DTW alignment)", test_mcd)

# ── Part 1 ─────────────────────────────────────────────────
print("\n[Part 1: STT]")

def test_denoiser():
    from src.part1.denoising import AudioDenoiser
    d = AudioDenoiser(method="spectral_subtraction", device="cpu")
    wav_np = np.random.randn(16000).astype(np.float32)
    result = d._spectral_subtraction(wav_np, 16000)
    assert len(result) == len(wav_np)
    assert result.dtype == np.float32
test("AudioDenoiser (spectral_subtraction)", test_denoiser)

def test_ngram():
    from src.part1.constrained_asr import NGramLM
    lm = NGramLM(order=3, smoothing_k=0.1)
    lm.train("the quick brown fox jumps over the lazy dog")
    lp = lm.log_prob("fox", ("quick", "brown"))
    assert lp < 0  # log prob is always negative
    terms = lm.get_technical_terms()
    assert isinstance(terms, list)
test("NGramLM (train + log_prob)", test_ngram)

def test_logit_bias():
    from src.part1.constrained_asr import LogitBiasProcessor, NGramLM
    lm = NGramLM(order=2)
    lm.train("stochastic cepstrum mfcc spectrogram formant")

    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [abs(hash(text)) % 1000]

    proc = LogitBiasProcessor(
        tokenizer=MockTokenizer(),
        ngram_lm=lm,
        technical_terms=["cepstrum", "stochastic"],
        bias_strength=2.5,
    )
    scores = torch.zeros(1, 1000)
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    out = proc(input_ids, scores)
    assert out.shape == scores.shape
test("LogitBiasProcessor", test_logit_bias)

def test_lid_model():
    from src.part1.lid_model import MultiHeadLID, ConvFeatureExtractor, MultiHeadLIDModel
    cnn = ConvFeatureExtractor(in_channels=80, out_channels=256)
    model = MultiHeadLIDModel(feature_dim=256, hidden_dim=256, num_heads=4, num_languages=2)
    mel = torch.randn(1, 80, 100)
    feat = cnn(mel)
    feat = feat.permute(0, 2, 1)
    logits, attn = model(feat, return_attention=True)
    assert logits.shape == (1, 100, 2)
test("LID Model (CNN + BiLSTM + MHA)", test_lid_model)

# ── Part 2 ─────────────────────────────────────────────────
print("\n[Part 2: IPA + Translation]")

def test_ipa_english():
    from src.part2.ipa_mapper import HinglishIPAMapper, ENGLISH_ACADEMIC_IPA
    mapper = HinglishIPAMapper(en_backend="espeak", code_switch_aware=True)
    ipa = mapper._word_to_ipa("cepstrum", "en")
    assert len(ipa) > 0
    assert ipa == ENGLISH_ACADEMIC_IPA["cepstrum"]
test("IPA mapper (English academic lookup)", test_ipa_english)

def test_ipa_hinglish():
    from src.part2.ipa_mapper import HinglishIPAMapper, HINGLISH_ROMAN_TO_IPA
    mapper = HinglishIPAMapper()
    ipa = mapper._word_to_ipa("nahi", "hi")
    assert ipa == HINGLISH_ROMAN_TO_IPA["nahi"]
test("IPA mapper (Hinglish Roman lookup)", test_ipa_hinglish)

def test_ipa_devanagari():
    from src.part2.ipa_mapper import HinglishIPAMapper
    mapper = HinglishIPAMapper()
    ipa = mapper._devanagari_to_ipa("क")
    assert "k" in ipa or len(ipa) > 0
test("IPA mapper (Devanagari rules)", test_ipa_devanagari)

def test_translator():
    from src.part2.lrl_translator import LRLTranslator
    t = LRLTranslator(target_language="maithili")
    result = t.translate(
        [{"start": 0.0, "end": 2.0, "text": "speech signal frequency"}]
    )
    assert "text" in result
    assert "segments" in result
    assert len(result["segments"]) == 1
    assert result["translation_coverage"] > 0
test("LRL Translator (Maithili lookup)", test_translator)

# ── Part 3 ─────────────────────────────────────────────────
print("\n[Part 3: Voice Cloning]")

def test_xvector_model():
    from src.part3.voice_embedding import XVectorExtractorModel
    model = XVectorExtractorModel(input_dim=80, embedding_dim=512)
    model.eval()  # Use eval mode to avoid BatchNorm single-sample issue
    with torch.no_grad():
        mel = torch.randn(2, 80, 300)  # batch=2 to satisfy BatchNorm
        emb = model(mel, return_embedding=True)
    assert emb.shape == (2, 512)
test("x-Vector model (TDNN forward pass)", test_xvector_model)

def test_stats_pooling():
    from src.part3.voice_embedding import StatisticsPooling
    pool = StatisticsPooling()
    x = torch.randn(2, 128, 50)
    out = pool(x)
    assert out.shape == (2, 256)  # [mean; std] concatenated
test("Statistics Pooling", test_stats_pooling)

def test_f0_extractor():
    from src.part3.prosody_warping import F0Extractor
    f0_ext = F0Extractor(sample_rate=16000)
    wav = torch.randn(1, 16000)
    f0, voiced = f0_ext._autocorr_f0(wav.squeeze().numpy())
    assert len(f0) > 0
    assert len(voiced) == len(f0)
test("F0 Extractor (autocorrelation fallback)", test_f0_extractor)

def test_dtw_warping():
    from src.part3.prosody_warping import ProsodyWarper
    warper = ProsodyWarper()
    src = np.random.randn(50, 2)
    tgt = np.random.randn(60, 2)
    warped = warper.apply_dtw_warping(src, tgt)
    assert warped.shape == (60, 2), f"Expected (60, 2), got {warped.shape}"
test("DTW Prosody Warping", test_dtw_warping)

def test_mock_synthesis():
    from src.part3.tts_synthesizer import LRLSynthesizer
    synth = LRLSynthesizer(device="cpu")
    synth._backend = "mock"
    audio = synth._mock_synthesis("hello world this is a test")
    assert audio.shape[0] == 1
    assert audio.shape[1] > 0
test("TTS Synthesizer (mock backend)", test_mock_synthesis)

def test_crossfade():
    from src.part3.tts_synthesizer import LRLSynthesizer
    synth = LRLSynthesizer(sample_rate=22050)
    chunks = [torch.randn(1, 22050) for _ in range(3)]
    result = synth._crossfade_stitch(chunks, crossfade_ms=50)
    assert result.shape[0] == 1
    assert result.shape[1] > 0
test("Cross-fade stitching", test_crossfade)

# ── Part 4 ─────────────────────────────────────────────────
print("\n[Part 4: Anti-Spoofing + Adversarial]")

def test_mfm():
    from src.part4.anti_spoofing import MaxFeatureMap2D
    mfm = MaxFeatureMap2D()
    x = torch.randn(2, 64, 10, 10)
    out = mfm(x)
    assert out.shape == (2, 32, 10, 10)
test("MaxFeatureMap2D (MFM activation)", test_mfm)

def test_lcnn_forward():
    from src.part4.anti_spoofing import LightCNN
    model = LightCNN(input_channels=1, n_lfcc=60)
    x = torch.randn(2, 1, 60, 300)  # (B, C, n_lfcc, T)
    logit = model(x)
    assert logit.shape == (2,), f"Expected shape (2,), got {logit.shape}"
test("LCNN forward pass", test_lcnn_forward)

def test_griffin_lim():
    from src.part4.adversarial import AdversarialAttacker
    attacker = AdversarialAttacker()
    mel = np.abs(np.random.randn(80, 100))
    wav = attacker._griffin_lim(mel, sr=16000, n_iter=5)
    assert len(wav) > 0
    assert np.isfinite(wav).all()
test("Griffin-Lim reconstruction", test_griffin_lim)

def test_fgsm_attack():
    from src.part4.adversarial import AdversarialAttacker
    attacker = AdversarialAttacker(attack_type="fgsm", target_snr_db=40.0)
    wav = torch.randn(1, 16000) * 0.1
    adv, snr = attacker._fgsm_attack(wav, lid_model=None, epsilon=0.01, sr=16000)
    assert adv.shape == wav.shape
    assert np.isfinite(snr)
test("FGSM attack (mel-domain)", test_fgsm_attack)

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 55)
passed = sum(results)
total = len(results)
pct = 100 * passed / total
print(f"Results: {passed}/{total} tests passed ({pct:.0f}%)")
if passed == total:
    print("✓ All tests passed. Ready to run pipeline.py!")
else:
    print("✗ Some tests failed. Check errors above before running pipeline.")
print("=" * 55 + "\n")
