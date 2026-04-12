#!/bin/bash
# Speech Assignment 2 — One-shot install script
# macOS Apple Silicon (M1/M2/M3/M4) | Python 3.12
#
# Usage (from project root, with venv active):
#   chmod +x install.sh && ./install.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Speech Assignment 2 — Install (macOS ARM)       ║"
echo "║  IIT Jodhpur | M25DE1047                         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── verify macOS ─────────────────────────────────────────────────
if [ "$(uname -s)" != "Darwin" ]; then
    echo "⚠  This script is for macOS. Detected: $(uname -s)"
    echo "   (This script is macOS only)"
    exit 1
fi

echo "Python:  $(python3 --version)"
echo "pip:     $(pip3 --version | cut -d' ' -f1-2)"
echo "Arch:    $(uname -m)"
echo ""

# ── Homebrew check ───────────────────────────────────────────────
echo "─── Checking Homebrew ──────────────────────────────"
if ! command -v brew &>/dev/null; then
    echo "✗  Homebrew not found. Install it first:"
    echo '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""
    echo "   Then re-run this script."
    exit 1
fi
echo "✓  Homebrew found: $(brew --version | head -1)"
echo ""

# ── system deps via Homebrew ─────────────────────────────────────
echo "─── System dependencies (Homebrew) ────────────────"
for pkg in ffmpeg espeak-ng; do
    if brew list "$pkg" &>/dev/null 2>&1; then
        echo "✓  $pkg already installed"
    else
        echo "Installing $pkg..."
        brew install "$pkg"
    fi
done
echo ""

# ── upgrade pip ──────────────────────────────────────────────────
echo "─── Upgrading pip ──────────────────────────────────"
pip3 install --upgrade pip setuptools wheel
echo ""

# ── PyTorch (MPS — Apple Silicon native) ─────────────────────────
echo "─── PyTorch for Apple Silicon (MPS) ────────────────"
echo "Note: macOS does NOT support CUDA. Use --device mps or --device cpu."
pip3 install torch torchaudio
echo ""

# ── audio & ML core ──────────────────────────────────────────────
echo "─── Core audio & ML packages ───────────────────────"
pip3 install \
    "numpy>=1.24.0" \
    "scipy>=1.10.0" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "audioread>=3.0.0"
echo ""

# ── ASR ──────────────────────────────────────────────────────────
echo "─── ASR packages ───────────────────────────────────"
pip3 install \
    openai-whisper \
    "transformers>=4.36.0" \
    "accelerate>=0.25.0" \
    "huggingface-hub>=0.20.0"
echo ""

# ── TTS (coqui-tts — community fork, py3.12 macOS ARM native) ────
echo "─── TTS backend (coqui-tts) ────────────────────────"
echo "Installing coqui-tts (replaces the abandoned 'TTS' package)..."
pip3 install coqui-tts || {
    echo "⚠  coqui-tts full install failed, trying minimal install..."
    pip3 install "coqui-tts" --no-deps 2>/dev/null || true
    pip3 install trainer coqpit anyascii gruut 2>/dev/null || true
    echo "   If synthesis fails, Meta MMS-TTS (via transformers) will be used automatically."
}
echo ""

# ── evaluation & utilities ───────────────────────────────────────
echo "─── Evaluation & utilities ─────────────────────────"
pip3 install jiwer matplotlib seaborn tqdm pandas pyyaml einops
echo ""

# ── deepfilternet (optional — needs Rust) ────────────────────────
echo "─── deepfilternet (optional neural denoiser) ────────"
if command -v cargo &>/dev/null || command -v rustup &>/dev/null; then
    echo "Rust found — installing deepfilternet..."
    pip3 install deepfilternet || echo "⚠  deepfilternet failed — spectral subtraction will be used instead"
else
    echo "⚠  Rust not found — skipping deepfilternet (spectral subtraction is the fallback)"
    echo "   To enable later: brew install rust && pip3 install deepfilternet"
fi
echo ""

# ── done ─────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════╗"
echo "║  Install complete!                               ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Next steps:                                     ║"
echo "║                                                  ║"
echo "║  1. python setup_check.py                        ║"
echo "║       Verify your environment                    ║"
echo "║                                                  ║"
echo "║  2. python prepare_data.py --input_m4a <file>   ║"
echo "║       Convert lecture M4A → WAV                  ║"
echo "║                                                  ║"
echo "║  3. Record 60s of your voice →                   ║"
echo "║       data/student_voice_ref.wav                 ║"
echo "║                                                  ║"
echo "║  4. python pipeline.py --device mps              ║"
echo "║       Run the pipeline (mps = Apple Silicon)     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
