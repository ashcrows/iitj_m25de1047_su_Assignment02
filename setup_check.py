"""
Environment compatibility checker for Speech Assignment 2.
Run this BEFORE pipeline.py to verify your setup.

Usage:
    python setup_check.py
"""

import sys
import shutil
import importlib
import platform

# ── colour helpers ────────────────────────────────────────────────
OK   = "✓"
WARN = "⚠"
FAIL = "✗"
INFO = "·"

def _green(s):  return f"\033[92m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _red(s):    return f"\033[91m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"

OS   = platform.system()    # 'Darwin' on macOS
ARCH = platform.machine()   # 'arm64' on Apple Silicon


def check_pkg(name, import_name=None, min_ver=None, required=True):
    import_name = import_name or name
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        ok  = True
        if min_ver and ver != "?" and ver < min_ver:
            icon = _yellow(WARN)
            note = f"(version {ver} < required {min_ver})"
        else:
            icon = _green(OK)
            note = f"({ver})"
        print(f"  {icon} {name:25s} {note}")
        return True
    except ImportError:
        tag  = _red(FAIL) if required else _yellow(WARN)
        note = "NOT INSTALLED — required" if required else "not installed — optional"
        print(f"  {tag} {name:25s} ({note})")
        return False


def check_binary(name, required=True):
    path = shutil.which(name)
    if path:
        print(f"  {_green(OK)} {name:25s} ({path})")
        return True
    tag  = _red(FAIL) if required else _yellow(WARN)
    note = "REQUIRED" if required else "optional"
    print(f"  {tag} {name:25s} ({note} — not found in PATH)")
    return False


def install_hint(pkg):
    """Return the correct install command for the current OS."""
    if OS == "Darwin":
        return f"brew install {pkg}"
    return f"brew install {pkg}"


def detect_tts_backend():
    try:
        from TTS.api import TTS  # noqa
        print(f"  {_green(OK)} TTS backend         (coqui-tts — best option)")
        return "coqui"
    except ImportError:
        pass
    try:
        from transformers import VitsModel  # noqa
        print(f"  {_yellow(WARN)} TTS backend         (Meta MMS-TTS via transformers — Hindi proxy)")
        return "mms"
    except ImportError:
        pass
    try:
        import torchaudio
        _ = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        print(f"  {_yellow(WARN)} TTS backend         (TorchAudio Tacotron2 — English only)")
        return "torchaudio"
    except Exception:
        pass
    print(f"  {_red(FAIL)} TTS backend         (NONE — install coqui-tts)")
    return "mock"


# ── Main ──────────────────────────────────────────────────────────

print()
print(_bold("=" * 56))
print(_bold(" Speech Assignment 2 — Environment Check"))
print(_bold(" IIT Jodhpur | M25DE1047"))
print(_bold("=" * 56))

# Python
major, minor = sys.version_info[:2]
py_ok = (major == 3 and minor >= 10)
icon  = _green(OK) if py_ok else _red(FAIL)
print(f"\n{icon} Python {major}.{minor}.{sys.version_info.micro}  ({sys.executable})")
if not py_ok:
    print(f"  {_red('Python 3.10+ required')}")

# Platform
print(f"{_green(OK)} Platform: {OS} {ARCH}")
if OS == "Darwin" and ARCH == "arm64":
    print(f"{_green(OK)} Apple Silicon detected — MPS acceleration available")
elif OS == "Darwin":
    print(f"{_yellow(WARN)} Intel Mac detected — CPU only (no MPS on Intel)")

# Compute device
print(f"\n{_bold('── Compute device ─────────────────────────────────')}")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  {_green(OK)} CUDA available   — use: --device cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  {_green(OK)} MPS available    — use: --device mps")
        print(f"  {INFO} macOS does NOT support CUDA — never use --device cuda")
    else:
        print(f"  {_yellow(WARN)} CPU only         — use: --device cpu")
except ImportError:
    print(f"  {_red(FAIL)} torch not installed yet")

print(f"\n{_bold('── Core packages ──────────────────────────────────')}")
core_ok = all([
    check_pkg("torch",      "torch",     "2.1.0"),
    check_pkg("torchaudio", "torchaudio","2.1.0"),
    check_pkg("numpy",      "numpy",     "1.24.0"),
    check_pkg("scipy",      "scipy",     "1.10.0"),
])

print(f"\n{_bold('── ASR (Part 1) ───────────────────────────────────')}")
check_pkg("openai-whisper", "whisper",      required=True)
check_pkg("transformers",   "transformers", "4.36.0", required=True)
check_pkg("accelerate",     "accelerate",   required=False)

print(f"\n{_bold('── Audio processing ───────────────────────────────')}")
check_pkg("librosa",   "librosa",   "0.10.0", required=True)
check_pkg("soundfile", "soundfile", required=True)
check_pkg("audioread", "audioread", required=True)

print(f"\n{_bold('── Denoising — Part 1 Task 1.3 (optional) ─────────')}")
dfn = check_pkg("deepfilternet", "df", required=False)
if not dfn:
    print(f"  {INFO} Spectral subtraction fallback will be used (built-in, no deps)")
    if OS == "Darwin":
        print(f"  {INFO} To enable DeepFilterNet: brew install rust && pip install deepfilternet")
    else:
        print(f"  {INFO} To enable DeepFilterNet: pip install deepfilternet")

print(f"\n{_bold('── TTS / Voice Cloning (Part 3) ───────────────────')}")
tts_backend = detect_tts_backend()
if tts_backend == "mock":
    print(f"  {INFO} Install coqui-tts:  pip install coqui-tts")

print(f"\n{_bold('── Evaluation & utilities ─────────────────────────')}")
check_pkg("jiwer",      "jiwer",      required=False)
check_pkg("matplotlib", "matplotlib", required=True)
check_pkg("seaborn",    "seaborn",    required=False)
check_pkg("tqdm",       "tqdm",       required=True)
check_pkg("pandas",     "pandas",     required=False)
check_pkg("einops",     "einops",     required=False)

print(f"\n{_bold('── System binaries ────────────────────────────────')}")
ffmpeg_ok = check_binary("ffmpeg", required=True)
espeak_ok = check_binary("espeak-ng", required=False)

if not ffmpeg_ok:
    print(f"  {INFO} Required for M4A → WAV conversion")
    if OS == "Darwin":
        print(f"  {INFO} Install: brew install ffmpeg")
    else:
        print(f"  {INFO} macOS install: brew install ffmpeg")

if not espeak_ok:
    if OS == "Darwin":
        print(f"  {INFO} Install: brew install espeak-ng")
    else:
        print(f"  {INFO} macOS install: brew install espeak-ng")

print(f"\n{_bold('── Data files ─────────────────────────────────────')}")
import os
data_files = {
    "data/original_segment.wav":  "run: python prepare_data.py --input_m4a <file>",
    "data/student_voice_ref.wav": "IMPORTANT — record your own 60s voice",
    "data/speech_syllabus.txt":   "auto-created by prepare_data.py",
    "data/maithili_corpus.json":  "auto-created by prepare_data.py",
}
for path, note in data_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {_green(OK)} {path:35s} ({size/1024:.0f} KB)")
    else:
        print(f"  {_yellow(WARN)} {path:35s} → {note}")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{_bold('=' * 56)}")

issues = []
if not core_ok:
    issues.append("Core packages missing — run: ./install.sh")
if not ffmpeg_ok:
    issues.append(f"ffmpeg missing — install with: {install_hint('ffmpeg')}")
if tts_backend == "mock":
    issues.append("No TTS backend — run: pip install coqui-tts")
if not os.path.exists("data/original_segment.wav"):
    issues.append("Audio not ready — run: python prepare_data.py --input_m4a <your_m4a>")
if not os.path.exists("data/student_voice_ref.wav"):
    issues.append("Voice reference missing — record 60s and save as data/student_voice_ref.wav")

if not issues:
    device_hint = "mps" if (OS == "Darwin" and ARCH == "arm64") else "cpu"
    print(_green(f" {OK} All systems ready!"))
    print(f"    Run: python pipeline.py --device {device_hint}")
else:
    print(_yellow(f" {WARN} {len(issues)} issue(s) to resolve:"))
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print(f"\n   Then re-run: python setup_check.py")

print(_bold("=" * 56))
print()
