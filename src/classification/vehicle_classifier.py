"""
Vehicle type classifier for the ALPR University Gate system.

Maps plate background color to vehicle type category based on
Indian motor vehicle regulations.
"""

from __future__ import annotations

# Color → Vehicle type mapping (Indian regulations)
COLOR_TO_TYPE: dict[str, str] = {
    "White":      "Private",
    "Yellow":     "Commercial",
    "Green":      "EV",
    "Red":        "Govt/Temp",
    "Blue":       "Diplomatic",
    "Black":      "Rental",
    "Army_Green": "Military",
    "Unknown":    "Unknown",
}

VALID_VEHICLE_TYPES = frozenset(COLOR_TO_TYPE.values())


class VehicleClassifier:
    """Classify vehicle type from plate color.

    Uses the COLOR_TO_TYPE mapping. Any color not in the mapping
    returns "Unknown".
    """

    def classify(self, color: str) -> str:
        """Map a plate color label to a vehicle type.

        Args:
            color: Color label from ColorClassifier (e.g. "White", "Yellow").

        Returns:
            Vehicle type string — always one of the values in VALID_VEHICLE_TYPES.
        """
        return COLOR_TO_TYPE.get(color, "Unknown")
