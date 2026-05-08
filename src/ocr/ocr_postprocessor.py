"""
OCR post-processor for the ALPR University Gate system.

Applies character-level corrections to raw OCR output to fix common
misreads caused by:
1. Visually similar characters (K↔M, 0↔O, 1↔I, etc.)
2. Bolt/screw interference on number plates
3. Font ambiguity in Indian license plates

Strategy:
- Apply position-aware corrections based on Indian plate format:
  XX00XX0000 — positions 0-1 are letters, 2-3 are digits,
                positions 4-5 are letters, 6-9 are digits
- Characters at digit positions: correct letter→digit confusions
- Characters at letter positions: correct digit→letter confusions
"""

from __future__ import annotations

import re

# Characters that look like digits but are letters
DIGIT_LIKE_LETTERS = {
    "O": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
    "T": "7",
}

# Characters that look like letters but are digits
LETTER_LIKE_DIGITS = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "6": "G",
    "2": "Z",
}

# Visually similar letter pairs in Indian plates (common OCR confusions)
# These are applied at letter positions only
SIMILAR_LETTERS = {
    # K and M are visually similar at low resolution
    # We don't auto-correct these as both are valid — majority voting handles it
}


def correct_ocr_text(raw_text: str) -> str:
    """Apply position-aware character corrections to raw OCR output.

    For Indian plate format XX00XX0000:
    - Positions 0-1: must be letters → convert digit-like chars to letters
    - Positions 2-3: must be digits → convert letter-like chars to digits
    - Positions 4-5: must be letters → convert digit-like chars to letters
    - Positions 6-9: must be digits → convert letter-like chars to digits

    Also handles BH series: BH00XX0000

    Args:
        raw_text: Raw OCR string (already stripped and uppercased).

    Returns:
        Corrected string. Returns original if length doesn't match expected format.
    """
    if not raw_text:
        return raw_text

    # Remove spaces, hyphens, dots that OCR sometimes inserts
    cleaned = raw_text.replace(" ", "").replace("-", "").replace(".", "").upper()

    if len(cleaned) != 10:
        return cleaned  # Can't apply positional correction

    chars = list(cleaned)

    # Positions 0-1: LETTERS (state code)
    for i in [0, 1]:
        if chars[i] in DIGIT_LIKE_LETTERS:
            chars[i] = DIGIT_LIKE_LETTERS[chars[i]]

    # Positions 2-3: DIGITS (district code)
    for i in [2, 3]:
        if chars[i] in LETTER_LIKE_DIGITS:
            chars[i] = LETTER_LIKE_DIGITS[chars[i]]
        # Also fix common digit confusions
        if chars[i] == "O":
            chars[i] = "0"
        if chars[i] == "I" or chars[i] == "L":
            chars[i] = "1"

    # Positions 4-5: LETTERS (series)
    for i in [4, 5]:
        if chars[i] in DIGIT_LIKE_LETTERS:
            chars[i] = DIGIT_LIKE_LETTERS[chars[i]]

    # Positions 6-9: DIGITS (registration number)
    for i in [6, 7, 8, 9]:
        if chars[i] in LETTER_LIKE_DIGITS:
            chars[i] = LETTER_LIKE_DIGITS[chars[i]]
        if chars[i] == "O":
            chars[i] = "0"
        if chars[i] == "I" or chars[i] == "L":
            chars[i] = "1"
        if chars[i] == "S":
            chars[i] = "5"
        if chars[i] == "B":
            chars[i] = "8"

    return "".join(chars)


def remove_noise_characters(raw_text: str) -> str:
    """Remove non-alphanumeric characters that OCR picks up from bolts/screws.

    Bolts, screws, stickers, and dirt on plates can cause OCR to read
    extra characters like '.', '/', '|', etc.

    Args:
        raw_text: Raw OCR string.

    Returns:
        String with only alphanumeric characters.
    """
    return re.sub(r"[^A-Z0-9]", "", raw_text.upper())
