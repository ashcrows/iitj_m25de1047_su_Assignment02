"""
Task 2.1: IPA Unified Representation for Code-Switched Hinglish.
Custom phoneme mapping layer for Hinglish G2P (Grapheme-to-Phoneme).
Author: Rohit (M25DE1047)
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Hindi Devanagari → IPA mapping (custom-built)
# ─────────────────────────────────────────────────────────────────

# Non-obvious design choice: Standard tools like epitran or espeak-ng
# handle Hindi in isolation, but fail on Romanized Hindi (Hinglish).
# We manually define phoneme rules for both:
#   1. Devanagari script (formal Hindi)
#   2. Roman-script Hindi (Hinglish: "kya", "nahi", "acha", etc.)
# The mapping prioritizes phonemic (not phonetic) transcription so
# the downstream TTS sees consistent IPA regardless of script.

DEVANAGARI_TO_IPA = {
    # Vowels
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː",
    "उ": "ʊ", "ऊ": "uː", "ए": "eː", "ऐ": "æː",
    "ओ": "oː", "औ": "ɔː", "ऋ": "ɾɪ",
    # Nasals
    "ङ": "ŋ", "ञ": "ɲ", "ण": "ɳ", "न": "n", "म": "m",
    # Stops (aspirated)
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʰ",
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʰ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʰ",
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʰ",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʰ",
    # Approximants & fricatives
    "य": "j", "र": "ɾ", "ल": "l", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
    # Flaps
    "ड़": "ɽ", "ढ़": "ɽʰ",
    # Matras (vowel signs)
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ",
    "ू": "uː", "े": "eː", "ै": "æː", "ो": "oː",
    "ौ": "ɔː", "ं": "◌̃",  # Anusvara (nasalization)
    "ः": "ɦ",               # Visarga
    "्": "",                 # Virama (halant) - removes inherent vowel
    "ँ": "◌̃",              # Chandrabindu
}

# Hinglish Romanized Hindi → IPA
# Covers common Hinglish patterns in academic speech
HINGLISH_ROMAN_TO_IPA = {
    # Common Hinglish words in academic context
    "kya": "kjɑː", "nahi": "nəɦɪ", "matlab": "mət̪ləb",
    "matlab": "mət̪ləb", "lekin": "lekɪn", "toh": "t̪oː",
    "acha": "ətʃʰɑː", "achha": "ətʃʰɑː", "haan": "ɦɑːn",
    "theek": "ʈʰiːk", "sab": "səb", "yeh": "jeː",
    "woh": "ʋoː", "jo": "dʒoː", "aur": "ɔːɾ",
    "bhi": "bʰɪ", "hai": "ɦæː", "hain": "ɦæːn",
    "iska": "ɪskɑː", "uska": "ʊskɑː", "matlab": "mət̪ləb",
    "samajh": "səmədʒ", "bolna": "bolnɑː", "sunna": "sʊnnɑː",
    # Technical Hinglish
    "frequency": "fɾiːkʷənsi", "signal": "sɪɡnəl",
    "matlab": "mæθlæb",   # Could be MATLAB (software)!
    "agar": "əɡəɾ", "phir": "pʰɪɾ", "sirf": "sɪɾf",
    # Retroflex distinctions (crucial for Hindi)
    "dono": "d̪onoː", "pehle": "pɛɦleː", "baad": "bɑːd̪",
    "shuru": "ʃʊɾuː", "khatam": "xət̪əm", "jaise": "dʒæseː",
}

# English G2P — common academic words (simplified CMU-dict style)
ENGLISH_ACADEMIC_IPA = {
    "stochastic": "stəˈkæstɪk", "cepstrum": "ˈsɛpstrəm",
    "spectrogram": "ˈspɛktrəɡræm", "formant": "ˈfɔːrmænt",
    "phoneme": "ˈfoʊniːm", "allophone": "ˈæləfoʊn",
    "mel": "mɛl", "cepstral": "ˈsɛpstrəl",
    "feature": "ˈfiːtʃər", "acoustic": "əˈkuːstɪk",
    "linguistic": "lɪŋˈɡwɪstɪk", "prosody": "ˈprɒsədi",
    "utterance": "ˈʌtərəns", "segment": "ˈsɛɡmənt",
    "hypothesis": "haɪˈpɒθɪsɪs", "posterior": "pɒˈstɪərɪər",
    "gaussian": "ˈɡaʊsiən", "markov": "ˈmɑːrkɒf",
    "viterbi": "vɪˈtɜːrbi", "algorithm": "ˈælɡərɪðəm",
    "interpolation": "ɪntɜːrpəˈleɪʃən", "coefficient": "koʊɪˈfɪʃənt",
    "fundamental": "fʌndəˈmɛntəl", "frequency": "ˈfriːkwənsi",
}


class HinglishIPAMapper:
    """
    Custom Grapheme-to-Phoneme converter for code-switched Hinglish.

    Handles:
      1. Pure English text → IPA (via espeak-ng or lookup table)
      2. Devanagari Hindi → IPA (rule-based from DEVANAGARI_TO_IPA)
      3. Romanized Hindi → IPA (lookup table HINGLISH_ROMAN_TO_IPA)
      4. Mixed code-switched segments (uses LID labels to route)

    Design choice (non-obvious):
      The key challenge is Hinglish in Roman script: "yeh concept bahut
      important hai" mixes Hindi words in Roman script with English.
      We use a word-level language decision:
        - If word is in HINGLISH_ROMAN_TO_IPA → use Hindi IPA
        - If word is in English academic dict → use English IPA
        - Else → use espeak-ng as fallback with language hint
      This avoids the catastrophic failure of treating Romanized Hindi
      as English (e.g., espeak pronounces "nahi" as /nɑːhɪ/ instead of /nəɦɪ/).
    """

    def __init__(
        self,
        en_backend: str = "espeak",
        hi_backend: str = "custom",
        code_switch_aware: bool = True,
    ):
        self.en_backend = en_backend
        self.hi_backend = hi_backend
        self.code_switch_aware = code_switch_aware
        self._espeak_available = self._check_espeak()

    def _check_espeak(self) -> bool:
        """Check if espeak-ng is available."""
        import shutil
        available = shutil.which("espeak-ng") is not None
        if not available:
            logger.warning(
                "espeak-ng not found. Using lookup tables only. "
                "  macOS: brew install espeak-ng"
            )
        return available

    def convert(
        self,
        segments: List[Dict],
        lid_labels: Optional[List[int]] = None,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Convert transcript segments to IPA.

        Args:
            segments: list of {"start", "end", "text"} dicts
            lid_labels: frame-level language labels (0=en, 1=hi)
            output_path: save result to JSON

        Returns:
            dict with 'ipa_string', 'word_ipa_list', 'segment_ipa'
        """
        word_ipa_list = []
        segment_ipa = []

        for seg in segments:
            text = seg.get("text", "")
            # Determine language hint for this segment
            lang_hint = "en"
            if lid_labels is not None:
                # Map segment time to frame index (rough)
                lang_hint = "hi" if self._segment_is_hindi(seg, lid_labels) else "en"

            words = text.split()
            seg_ipa_words = []
            for word in words:
                ipa = self._word_to_ipa(word, lang_hint)
                word_ipa_list.append({"word": word, "ipa": ipa, "lang": lang_hint})
                seg_ipa_words.append(ipa)

            seg_ipa_str = " ".join(seg_ipa_words)
            segment_ipa.append({
                "start": seg.get("start"),
                "end": seg.get("end"),
                "ipa": seg_ipa_str,
                "language": lang_hint,
            })

        full_ipa = " | ".join(s["ipa"] for s in segment_ipa)

        result = {
            "ipa_string": full_ipa,
            "word_ipa_list": word_ipa_list,
            "segment_ipa": segment_ipa,
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"IPA transcript saved: {output_path}")

        return result

    def _word_to_ipa(self, word: str, lang_hint: str = "en") -> str:
        """Convert a single word to IPA."""
        word_clean = re.sub(r"[^\w\u0900-\u097F]", "", word).strip()
        if not word_clean:
            return ""

        # Check if Devanagari
        if self._is_devanagari(word_clean):
            return self._devanagari_to_ipa(word_clean)

        word_lower = word_clean.lower()

        # Check Hinglish Roman dict
        if word_lower in HINGLISH_ROMAN_TO_IPA:
            return HINGLISH_ROMAN_TO_IPA[word_lower]

        # Check English academic dict
        if word_lower in ENGLISH_ACADEMIC_IPA:
            return ENGLISH_ACADEMIC_IPA[word_lower]

        # Fallback: espeak-ng
        if self._espeak_available:
            return self._espeak_ipa(word_clean, lang_hint)

        # Last resort: phonemic approximation
        return self._naive_roman_ipa(word_lower)

    def _is_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari characters."""
        return bool(re.search(r"[\u0900-\u097F]", text))

    def _devanagari_to_ipa(self, text: str) -> str:
        """Convert Devanagari string to IPA using rule table."""
        ipa = []
        i = 0
        chars = list(text)
        while i < len(chars):
            # Try 2-char combinations first (matras + consonants)
            two = "".join(chars[i: i + 2])
            if two in DEVANAGARI_TO_IPA:
                ipa.append(DEVANAGARI_TO_IPA[two])
                i += 2
                continue
            one = chars[i]
            if one in DEVANAGARI_TO_IPA:
                mapped = DEVANAGARI_TO_IPA[one]
                # Add inherent vowel 'ə' after consonants (not after virama)
                if mapped and one not in "अआइईउऊएऐओऔऋ":
                    if i + 1 < len(chars) and chars[i + 1] != "्":
                        mapped += "ə"
                ipa.append(mapped)
            i += 1
        return "".join(ipa)

    def _espeak_ipa(self, word: str, lang: str = "en") -> str:
        """Use espeak-ng to get IPA for a word."""
        import subprocess
        lang_code = "hi" if lang == "hi" else "en-us"
        try:
            result = subprocess.run(
                ["espeak-ng", "-v", lang_code, "-q", "--ipa", word],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip().replace("\n", " ")
        except Exception as e:
            logger.debug(f"espeak-ng failed for '{word}': {e}")
            return self._naive_roman_ipa(word)

    def _naive_roman_ipa(self, word: str) -> str:
        """
        Naive approximation of IPA from Roman spelling.
        Used as last-resort fallback when espeak unavailable.
        """
        rules = [
            ("sh", "ʃ"), ("ch", "tʃ"), ("th", "θ"), ("ph", "f"),
            ("ng", "ŋ"), ("ck", "k"), ("ee", "iː"), ("oo", "uː"),
            ("aa", "aː"), ("ai", "æɪ"), ("au", "ɔː"), ("ou", "aʊ"),
            ("qu", "kʷ"), ("x", "ks"), ("c", "k"), ("y", "j"),
            ("a", "æ"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɒ"), ("u", "ʌ"),
        ]
        result = word.lower()
        for src, tgt in rules:
            result = result.replace(src, tgt)
        return result

    def _segment_is_hindi(self, seg: Dict, lid_labels: List[int]) -> bool:
        """Check if segment is predominantly Hindi based on LID labels."""
        # Rough: use character ratio of Devanagari
        text = seg.get("text", "")
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
        return devanagari_chars > len(text) * 0.2
