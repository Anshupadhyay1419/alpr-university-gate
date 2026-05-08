"""
Indian license plate validator for the ALPR University Gate system.

Validates OCR output against Indian plate formats including BH series.

Supported formats:
  - Standard: XX00XX0000  (e.g. KA19TR0234)
  - BH series: BH00XX0000 (e.g. BH01AB1234)

Regex: ^(([A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4})|(BH[0-9]{2}[A-Z]{2}[0-9]{4}))$
"""

from __future__ import annotations

import re

PLATE_PATTERN = re.compile(
    r"^(([A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4})|(BH[0-9]{2}[A-Z]{2}[0-9]{4}))$"
)

# Also accept 9-char format: XX0XX0000 (single district digit, e.g. DL7CD5017)
PLATE_PATTERN_9 = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
)


class PlateValidator:
    """Validate OCR text against Indian license plate formats.

    Returns the normalized plate string and series type on match,
    or (None, None) if the string does not match any valid format.
    """

    def validate(self, ocr_text: str) -> tuple[str | None, str | None]:
        """Validate and classify an OCR string."""
        if not ocr_text:
            return (None, None)

        # Normalize: strip whitespace, uppercase, remove hyphens/spaces/dots
        normalized = ocr_text.strip().upper().replace(" ", "").replace("-", "").replace(".", "").replace("/", "")

        # Try direct match first
        if PLATE_PATTERN.match(normalized):
            series_type = "BH" if normalized.startswith("BH") else "normal"
            return (normalized, series_type)

        # Try OCR correction: fix common substitutions at known positions
        # Indian plate format: XX00XX0000 (10 chars)
        # Position 0,1: letters — O→0 wrong, 0→O fix
        # Position 2,3: digits  — O→0 fix
        # Position 4,5: letters — O→0 wrong, 0→O fix
        # Position 6-9: digits  — O→0 fix
        if len(normalized) == 10:
            chars = list(normalized)
            # Positions 0,1 must be letters
            for i in [0, 1]:
                if chars[i] == '0': chars[i] = 'O'
                if chars[i] == '1': chars[i] = 'I'
            # Positions 2,3 must be digits
            for i in [2, 3]:
                if chars[i] == 'O': chars[i] = '0'
                if chars[i] == 'I' or chars[i] == 'L': chars[i] = '1'
            # Positions 4,5 must be letters
            for i in [4, 5]:
                if chars[i] == '0': chars[i] = 'O'
                if chars[i] == '1': chars[i] = 'I'
            # Positions 6-9 must be digits
            for i in [6, 7, 8, 9]:
                if chars[i] == 'O': chars[i] = '0'
                if chars[i] == 'I' or chars[i] == 'L': chars[i] = '1'
                if chars[i] == 'S': chars[i] = '5'
                if chars[i] == 'B': chars[i] = '8'
                if chars[i] == 'Z': chars[i] = '2'
            corrected = "".join(chars)
            if PLATE_PATTERN.match(corrected):
                series_type = "BH" if corrected.startswith("BH") else "normal"
                return (corrected, series_type)

        # If 9 chars, OCR may have missed first character — try common state prefixes
        if len(normalized) == 9:
            for prefix in ['K', 'M', 'D', 'T', 'G', 'A', 'H', 'R', 'U', 'B', 'X']:
                candidate = prefix + normalized
                chars = list(candidate)
                # Apply same corrections
                for i in [0, 1]:
                    if chars[i] == '0': chars[i] = 'O'
                for i in [2, 3]:
                    if chars[i] == 'O': chars[i] = '0'
                for i in [4, 5]:
                    if chars[i] == '0': chars[i] = 'O'
                for i in [6, 7, 8, 9]:
                    if chars[i] == 'O': chars[i] = '0'
                    if chars[i] == 'S': chars[i] = '5'
                candidate = "".join(chars)
                if PLATE_PATTERN.match(candidate):
                    series_type = "BH" if candidate.startswith("BH") else "normal"
                    return (candidate, series_type)

        # Accept flexible Indian plate format (8-11 chars)
        # Covers non-standard formats: DL7CD5017, DL3CBJ1384, DL2CAT4762
        if PLATE_PATTERN_9.match(normalized) and 8 <= len(normalized) <= 11:
            series_type = "BH" if normalized.startswith("BH") else "normal"
            return (normalized, series_type)

        return (None, None)
