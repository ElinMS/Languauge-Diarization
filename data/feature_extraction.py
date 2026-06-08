"""
feature_extraction.py
─────────────────────
Converts raw waveforms → log-mel filterbank features used as
the Conformer encoder's input.

Shape convention throughout the project:
  (Batch, Time-frames, n_mels)  →  "BTF"
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


class LogMelFrontend(nn.Module):
    """
    Learnable-free log-mel frontend.

    Pipeline
    --------
    waveform  →  MelSpectrogram  →  log(x + ε)  →  per-utterance normalisation
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,       # 10 ms
        win_length: int = 400,       # 25 ms
        f_min: float = 0.0,
        f_max: float = 8_000.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : (B, T_samples)  –  float32, values in [-1, 1]

        Returns
        -------
        feats : (B, T_frames, n_mels)
        """
        # (B, n_mels, T_frames)
        mel = self.mel_spec(waveform)
        mel = self.amplitude_to_db(mel)

        # → (B, T_frames, n_mels)
        feats = mel.transpose(1, 2)

        if self.normalize:
            # per-utterance mean-variance normalisation
            mean = feats.mean(dim=1, keepdim=True)
            std  = feats.std(dim=1, keepdim=True).clamp(min=1e-5)
            feats = (feats - mean) / std

        return feats


class SpecAugment(nn.Module):
    """
    SpecAugment (Park et al. 2019) applied in training only.
    Masks random time steps and frequency bands.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,    # max frequency bins to mask
        time_mask_param: int = 100,   # max time frames to mask
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq = num_freq_masks
        self.num_time = num_time_masks

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats : (B, T_frames, n_mels)
        """
        # torchaudio masking expects (B, n_mels, T) — so we transpose
        x = feats.transpose(1, 2)   # (B, n_mels, T)
        for _ in range(self.num_freq):
            x = self.freq_masking(x)
        for _ in range(self.num_time):
            x = self.time_masking(x)
        return x.transpose(1, 2)    # back to (B, T, n_mels)


class WavegramNet(nn.Module):
    """
    Extracts a learned 1D Wavegram from a raw 1D audio waveform.
    Matches the SW-WaveNet paper specification:
    A single 1D Convolutional layer with K=1024, Cout=128, and D=1.
    Stride is set to 160 to match log-mel spectrogram hop_length (10ms @ 16kHz).
    """
    def __init__(self, out_channels=128, stride=160, kernel_size=1024):
        super(WavegramNet, self).__init__()
        # K=1024, Cout=128, D=1. Stride adjusted to 160 to match mel-spec frames.
        self.conv = nn.Conv1d(1, out_channels, kernel_size=kernel_size, stride=stride)
        
        # We need padding to align the time dimension with mel-spec
        # The mel-spec uses center=True and reflect padding.
        # For simplicity, we can dynamically pad inside the forward pass or use fixed padding.

    def forward(self, x):
        """
        Args:
            x: Raw waveform tensor of shape [Batch, Time]
        Returns:
            wavegram: Learned representation of shape [Batch, Time', 128]
        """
        # x is (Batch, Time), conv1d needs (Batch, Channels, Time)
        x = x.unsqueeze(1)
        
        # Calculate padding to match the number of frames of mel-spectrogram
        # For a signal of length L, mel-spec with center=True, hop=160 produces frames: L // 160 + 1
        # Conv1d with stride=160 produces frames: (L + 2*P - K) // 160 + 1
        # To make them match: 2*P - K = 0  => P = K // 2
        pad = self.conv.kernel_size[0] // 2
        x = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
        
        wavegram = self.conv(x)
        
        # wavegram is (Batch, out_channels, Time')
        # transpose to (Batch, Time', out_channels)
        return wavegram.transpose(1, 2)


class DualFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float = 8_000.0,
        normalize: bool = True,
        wavegram_channels: int = 128,
        wavegram_kernel: int = 1024,
    ):
        super().__init__()
        self.mel_frontend = LogMelFrontend(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            normalize=normalize,
        )
        self.wavegram_net = WavegramNet(
            out_channels=wavegram_channels,
            stride=hop_length,
            kernel_size=wavegram_kernel,
        )
        self.normalize = normalize

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T_samples)
        Returns concatenated features: (B, T_frames, n_mels + wavegram_channels)
        """
        mel_feats = self.mel_frontend(waveform)           # (B, T_frames, n_mels)
        wave_feats = self.wavegram_net(waveform)          # (B, T_frames_w, wave_channels)
        
        # Because of minor padding differences, ensure time dims match
        min_len = min(mel_feats.size(1), wave_feats.size(1))
        mel_feats = mel_feats[:, :min_len, :]
        wave_feats = wave_feats[:, :min_len, :]
        
        if self.normalize:
            mean = wave_feats.mean(dim=1, keepdim=True)
            std  = wave_feats.std(dim=1, keepdim=True).clamp(min=1e-5)
            wave_feats = (wave_feats - mean) / std
            
        combined = torch.cat([mel_feats, wave_feats], dim=-1) # (B, min_len, n_mels + wavegram_channels)
        return combined
