import torch, json
from pathlib import Path

class Normalizer:
    """Per‑variable z‑score normalisation (Algorithm 1)."""
    def __init__(self, stats_path: Path):
        self.stats = json.load(open(stats_path))
    def encode(self, x, key):
        μ, σ = self.stats[key]["mean"], self.stats[key]["std"]
        return (x - μ) / σ
    def decode(self, x, key):
        μ, σ = self.stats[key]["mean"], self.stats[key]["std"]
        return x * σ + μ