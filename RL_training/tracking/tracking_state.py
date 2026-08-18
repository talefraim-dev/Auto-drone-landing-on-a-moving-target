from enum import Enum


class TrackingMode(Enum):
    MATCH = "MATCH"
    PRED = "PRED"
    LOST = "LOST"
