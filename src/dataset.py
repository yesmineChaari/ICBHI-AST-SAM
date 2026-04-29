"""
Changes :
  - Added Gaussian noise augmentation (in train only)
  - Added SpecAugment: frequency masking + time masking 
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset


def add_gaussian_noise(wav: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """Add white Gaussian noise at a target SNR in dB."""
    rms_signal = np.sqrt(np.mean(wav ** 2)) + 1e-9
    rms_noise  = rms_signal / (10 ** (snr_db / 20))
    noise      = np.random.randn(len(wav)) * rms_noise
    return (wav + noise).astype(np.float32)


class ASTDataset(Dataset):
    """
    Same interface as the original ASTDataset.
    New: `augment` flag (defaults to the value of `train`).
    When augment=True, applies Gaussian noise + SpecAugment.
    """

    # ── SpecAugment params (conservative) ────────────────────────────────────
    FREQ_MASK_F = 15   # max consecutive mel-freq bins to zero out
    TIME_MASK_T = 60   # max consecutive time frames to zero out

    # ── Noise params ─────────────────────────────────────────────────────────
    P_NOISE  = 0.30          # probability of adding noise per sample
    SNR_LOW  = 25.0          # audible but mild noise
    SNR_HIGH = 40.0          # very mild noise

    SR = 16_000

    def __init__(self, X, y, device_ids, processor, train: bool = True, augment=None):
        self.X          = X
        self.y          = y
        self.device_ids = device_ids
        self.processor  = processor
        self.augment    = train if augment is None else augment

    def _apply_spec_augment(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (T, F) — AST input features (time × mel-freq)
        Applies one freq mask and one time mask in-place.
        """
        T, F = features.shape

        # Frequency masking
        f  = random.randint(0, self.FREQ_MASK_F)
        f0 = random.randint(0, max(F - f, 0))
        features[:, f0: f0 + f] = 0.0

        # Time masking
        t  = random.randint(0, self.TIME_MASK_T)
        t0 = random.randint(0, max(T - t, 0))
        features[t0: t0 + t, :] = 0.0

        return features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        wav = self.X[idx].copy()  

        # ── Waveform augmentation (Gaussian noise) ────────────────────────────
        if self.augment and random.random() < self.P_NOISE:
            snr = random.uniform(self.SNR_LOW, self.SNR_HIGH)
            wav = add_gaussian_noise(wav, snr_db=snr)

        # ── AST feature extraction ────────────────────
        features = self.processor(
            [wav],
            sampling_rate=self.SR,
            return_tensors="pt",
            padding=True,
        )
        input_values = features["input_values"].squeeze(0) 

        # ── SpecAugment ──────────────────────────
        if self.augment and random.random() < 0.5:
            input_values = self._apply_spec_augment(input_values)

        label     = int(self.y[idx])
        device_id = int(self.device_ids[idx])

        return input_values, label, device_id