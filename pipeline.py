"""
Speech Understanding - Programming Assignment 2
Main Pipeline: Code-Switched STT → Phonetic Mapping → LRL TTS → Anti-Spoofing
Author: Ashish Siha (M25DE1047), MTech, IIT Jodhpur
"""

import argparse
import logging
import os
import sys

import torch

# Local imports
from src.part1.denoising import AudioDenoiser
from src.part1.lid_model import MultiHeadLID
from src.part1.constrained_asr import ConstrainedASR
from src.part2.ipa_mapper import HinglishIPAMapper
from src.part2.lrl_translator import LRLTranslator
from src.part3.voice_embedding import VoiceEmbeddingExtractor
from src.part3.prosody_warping import ProsodyWarper
from src.part3.tts_synthesizer import LRLSynthesizer
from src.part4.anti_spoofing import AntiSpoofingCM
from src.part4.adversarial import AdversarialAttacker
from src.utils.audio_utils import load_audio, save_audio, get_audio_segment
from src.utils.evaluation import evaluate_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")



def _best_device() -> str:
    """
    Auto-detect the best available compute device.
      - 'cuda' on Linux/Windows with NVIDIA GPU
      - 'mps'  on macOS Apple Silicon (M1/M2/M3/M4)
      - 'cpu'  everywhere else
    Note: macOS does NOT support CUDA. Passing --device cuda on a Mac
    will raise a RuntimeError; use --device mps or --device cpu instead.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Code-Switched STT → LRL TTS Pipeline"
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        default="data/original_segment.wav",
        help="Path to source lecture audio (WAV)",
    )
    parser.add_argument(
        "--voice_ref",
        type=str,
        default="data/student_voice_ref.wav",
        help="Path to 60s student voice reference (WAV)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Directory for all outputs",
    )
    parser.add_argument(
        "--target_lrl",
        type=str,
        default="maithili",
        choices=["maithili", "santhali", "gondi"],
        help="Target Low-Resource Language",
    )
    parser.add_argument(
        "--segment_duration",
        type=float,
        default=600.0,
        help="Duration in seconds to process (default 10 min)",
    )
    parser.add_argument(
        "--run_part",
        type=str,
        default="all",
        choices=["all", "part1", "part2", "part3", "part4"],
        help="Run specific part only",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_best_device(),
        help="Device: cuda | mps | cpu  (auto-detected if not specified)",
    )
    parser.add_argument(
        "--syllabus_path",
        type=str,
        default="data/speech_syllabus.txt",
        help="Path to course syllabus for N-gram LM",
    )
    return parser.parse_args()


def run_part1(args, audio_path: str) -> dict:
    """
    Part I: Robust Code-Switched Transcription (STT)
    Returns dict with transcript, LID labels, segment info.
    """
    logger.info("=" * 60)
    logger.info("PART I: Code-Switched Transcription")
    logger.info("=" * 60)

    # Task 1.3: Denoising first
    logger.info("[Task 1.3] Denoising & normalization...")
    denoiser = AudioDenoiser(method="spectral_subtraction", device=args.device)
    denoised_path = os.path.join(args.output_dir, "denoised_segment.wav")
    denoiser.process(audio_path, denoised_path)
    logger.info(f"Denoised audio saved to: {denoised_path}")

    # Task 1.1: Frame-level Language Identification
    logger.info("[Task 1.1] Running Multi-Head LID...")
    lid_model = MultiHeadLID(
        num_languages=2,      # English, Hindi
        feature_dim=80,       # Mel filterbank features
        hidden_dim=256,
        num_heads=4,
        device=args.device,
    )
    lid_results = lid_model.predict(denoised_path)
    lid_model.save_results(lid_results, os.path.join(args.output_dir, "lid_results.json"))
    logger.info(f"LID F1-score (English): {lid_results['f1_en']:.4f}")
    logger.info(f"LID F1-score (Hindi):   {lid_results['f1_hi']:.4f}")

    # Task 1.2: Constrained Decoding with N-gram LM
    logger.info("[Task 1.2] Constrained Beam Search ASR...")
    asr = ConstrainedASR(
        model_name="openai/whisper-large-v3",
        syllabus_path=args.syllabus_path,
        ngram_order=3,
        logit_bias_strength=2.5,
        device=args.device,
    )
    transcript = asr.transcribe(
        denoised_path,
        lid_results=lid_results,
        output_path=os.path.join(args.output_dir, "transcript.json"),
    )
    logger.info(f"Transcript (first 200 chars): {transcript['text'][:200]}...")

    return {
        "denoised_path": denoised_path,
        "transcript": transcript,
        "lid_results": lid_results,
    }


def run_part2(args, part1_results: dict) -> dict:
    """
    Part II: Phonetic Mapping & Translation to LRL
    Returns IPA string and translated LRL text/phonemes.
    """
    logger.info("=" * 60)
    logger.info("PART II: Phonetic Mapping & Translation")
    logger.info("=" * 60)

    transcript = part1_results["transcript"]

    # Task 2.1: IPA conversion for code-switched Hinglish
    logger.info("[Task 2.1] Converting Hinglish to IPA...")
    ipa_mapper = HinglishIPAMapper(
        en_backend="espeak",
        hi_backend="custom",          # Custom Devanagari phoneme rules
        code_switch_aware=True,
    )
    ipa_result = ipa_mapper.convert(
        transcript["segments"],
        lid_labels=part1_results["lid_results"]["frame_labels"],
        output_path=os.path.join(args.output_dir, "ipa_transcript.json"),
    )
    logger.info(f"IPA (sample): {ipa_result['ipa_string'][:100]}")

    # Task 2.2: Semantic translation to target LRL
    logger.info(f"[Task 2.2] Translating to {args.target_lrl}...")
    translator = LRLTranslator(
        target_language=args.target_lrl,
        corpus_path=f"data/{args.target_lrl}_corpus.json",
        fallback_strategy="phonetic_transfer",  # When no translation exists
    )
    lrl_result = translator.translate(
        transcript["segments"],
        ipa_result,
        output_path=os.path.join(args.output_dir, f"lrl_translation_{args.target_lrl}.json"),
    )
    logger.info(f"LRL translation sample: {str(lrl_result['text'])[:200]}")

    return {
        "ipa_result": ipa_result,
        "lrl_result": lrl_result,
    }


def run_part3(args, part1_results: dict, part2_results: dict) -> dict:
    """
    Part III: Zero-Shot Cross-Lingual Voice Cloning
    Returns path to synthesized LRL audio.
    """
    logger.info("=" * 60)
    logger.info("PART III: Zero-Shot Voice Cloning")
    logger.info("=" * 60)

    # Task 3.1: Extract speaker embedding from student voice
    logger.info("[Task 3.1] Extracting d-vector speaker embedding...")
    embedder = VoiceEmbeddingExtractor(
        model_type="xvector",       # x-vector architecture (TDNN-based)
        embedding_dim=512,
        device=args.device,
    )
    speaker_embedding = embedder.extract(
        args.voice_ref,
        output_path=os.path.join(args.output_dir, "speaker_embedding.pt"),
    )
    logger.info(f"Speaker embedding shape: {speaker_embedding.shape}")

    # Task 3.2: Prosody Warping with DTW
    logger.info("[Task 3.2] Extracting prosody & applying DTW warping...")
    prosody_warper = ProsodyWarper(
        f0_extractor="pyin",        # PYIN algorithm for robust F0
        energy_frame_size=512,
        dtw_metric="euclidean",
        device=args.device,
    )
    warped_prosody = prosody_warper.warp(
        source_path=part1_results["denoised_path"],
        output_path=os.path.join(args.output_dir, "warped_prosody.pt"),
    )
    logger.info("Prosody warping complete.")

    # Task 3.3: Synthesize with VITS
    logger.info("[Task 3.3] Synthesizing LRL speech with VITS...")
    synthesizer = LRLSynthesizer(
        model_name="vits",
        target_language=args.target_lrl,
        sample_rate=22050,           # 22.05kHz requirement
        device=args.device,
    )
    output_audio_path = os.path.join(args.output_dir, "output_LRL_cloned.wav")
    synthesizer.synthesize(
        lrl_text=part2_results["lrl_result"],
        speaker_embedding=speaker_embedding,
        prosody=warped_prosody,
        output_path=output_audio_path,
    )
    logger.info(f"Synthesized audio saved: {output_audio_path}")

    return {
        "speaker_embedding": speaker_embedding,
        "warped_prosody": warped_prosody,
        "output_audio_path": output_audio_path,
    }


def run_part4(args, part1_results: dict, part3_results: dict) -> dict:
    """
    Part IV: Adversarial Robustness & Anti-Spoofing
    Returns EER, adversarial epsilon, confusion matrices.
    """
    logger.info("=" * 60)
    logger.info("PART IV: Anti-Spoofing & Adversarial Robustness")
    logger.info("=" * 60)

    # Task 4.1: Anti-Spoofing CM with LFCC
    logger.info("[Task 4.1] Training Anti-Spoofing Countermeasure...")
    cm = AntiSpoofingCM(
        feature_type="lfcc",        # LFCC features
        num_ceps=60,
        classifier="lcnn",          # Light CNN architecture
        device=args.device,
    )
    cm.train(
        bona_fide_path=args.voice_ref,
        spoof_path=part3_results["output_audio_path"],
        checkpoint_path=os.path.join(args.output_dir, "cm_model.pt"),
    )
    eer = cm.evaluate(
        bona_fide_path=args.voice_ref,
        spoof_path=part3_results["output_audio_path"],
        output_path=os.path.join(args.output_dir, "cm_evaluation.json"),
    )
    logger.info(f"Anti-Spoofing EER: {eer:.4f} ({eer*100:.2f}%)")

    # Task 4.2: FGSM Adversarial Perturbation on LID
    logger.info("[Task 4.2] FGSM adversarial attack on LID...")
    attacker = AdversarialAttacker(
        attack_type="fgsm",
        target_snr_db=40.0,         # Inaudible constraint: SNR > 40dB
        device=args.device,
    )
    adv_results = attacker.find_min_epsilon(
        audio_path=part1_results["denoised_path"],
        lid_model_path=os.path.join(args.output_dir, "lid_results.json"),
        target_flip="hindi_to_english",
        output_path=os.path.join(args.output_dir, "adversarial_results.json"),
    )
    logger.info(f"Minimum epsilon for LID flip: {adv_results['min_epsilon']:.6f}")
    logger.info(f"Achieved SNR: {adv_results['snr_db']:.2f} dB")

    return {
        "eer": eer,
        "adv_results": adv_results,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Speech Assignment 2 Pipeline Starting")
    logger.info(f"Device: {args.device}")
    logger.info(f"Target LRL: {args.target_lrl}")
    logger.info("=" * 60)

    # Prepare input: extract 10-minute segment
    logger.info(f"Loading audio: {args.input_audio}")
    segment_path = os.path.join(args.output_dir, "original_segment.wav")
    get_audio_segment(
        args.input_audio,
        segment_path,
        start_sec=0.0,
        duration_sec=args.segment_duration,
    )

    results = {}

    if args.run_part in ("all", "part1"):
        results["part1"] = run_part1(args, segment_path)

    if args.run_part in ("all", "part2"):
        if "part1" not in results:
            raise ValueError("Part 2 requires Part 1 results. Run with --run_part all or part1 first.")
        results["part2"] = run_part2(args, results["part1"])

    if args.run_part in ("all", "part3"):
        if "part1" not in results or "part2" not in results:
            raise ValueError("Part 3 requires Parts 1 and 2 results.")
        results["part3"] = run_part3(args, results["part1"], results["part2"])

    if args.run_part in ("all", "part4"):
        if "part1" not in results or "part3" not in results:
            raise ValueError("Part 4 requires Parts 1 and 3 results.")
        results["part4"] = run_part4(args, results["part1"], results["part3"])

    # Final evaluation report
    if args.run_part == "all":
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION SUMMARY")
        logger.info("=" * 60)
        evaluate_pipeline(results, args.output_dir)

    logger.info("Pipeline complete. All outputs saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
