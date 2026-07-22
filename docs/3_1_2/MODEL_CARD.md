# Model Card

    Frozen chain: Presence Gate -> Smoke R1 Top-1 -> SAM-HQ bbox prompt -> coarse bbox runtime fallback.

    Official sealed metrics are the source of truth:
    {
  "Positive BBox@0.5": 0.5173,
  "Positive BBox-Mask E2E": 0.2733,
  "Natural MVP Mixed": 0.5322,
  "Balanced v2 MVP Mixed": 0.4953,
  "Positive P95 ms": 220.15,
  "Natural P95 ms": 220.18
}

    Production quality target met: false.
    Mask quality limited: true.
