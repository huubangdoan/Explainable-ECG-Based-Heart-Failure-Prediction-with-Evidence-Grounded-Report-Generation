"""Independent PyTorch implementation of the pinned PTB-XL Wang configuration.

Research attribution: Strodthoff et al., https://arxiv.org/abs/2004.13701
Architectural reference: helme/ecg_ptbxl_benchmarking, commit
cdbf4e66d7e57d9b6a2657b6024716212b8d0afa, resnet1d.py/basic_conv1d.py.
This file does not import or vendor the historical Fastai implementation.

Intent: preserve that configuration's widths, residual topology and head.
Status: independently implemented; CUDA execution is target-only validation.
Next: compare identical-cohort validation results, not historical headline scores.
Confidence: architecture reviewed statically; trained behavior unmeasured here.
"""

from __future__ import annotations

import torch
from torch import nn


def _scaled_kernel(base: int, fs: int) -> int:
    value = int(round(base * fs / 100))
    return value if value % 2 else value + 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, int], stride: int):
        super().__init__()
        first, second = kernels
        self.branch = nn.Sequential(
            nn.Conv1d(channels, channels, first, stride=stride, padding=first // 2, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, second, padding=second // 2, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.shortcut = (nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv1d(channels, channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(channels),
        ))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, signal):
        return self.activation(self.branch(signal) + self.shortcut(signal))


class WangResNet1d(nn.Module):
    def __init__(self, fs: int, num_classes: int = 5):
        super().__init__()
        if fs not in (100, 200, 500):
            raise ValueError("Only the frozen 100, 200 and 500 Hz experiments are supported")
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be positive")
        stem_kernel = _scaled_kernel(7, fs)
        kernels = (_scaled_kernel(5, fs), _scaled_kernel(3, fs))
        self.features = nn.Sequential(
            nn.Conv1d(12, 128, stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            *(ResidualBlock(128, kernels, stride) for stride in (1, 2, 2)),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(256), nn.Dropout(0.25), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.BatchNorm1d(128), nn.Dropout(0.5), nn.Linear(128, num_classes),
        )
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.architecture_details = {
            "name": "resnet1d_wang_modern_port", "sampling_frequency": fs,
            "stem_kernel": stem_kernel, "block_kernels": list(kernels),
            "block_strides": [1, 2, 2], "width": 128, "blocks": 3,
            "convolutions_per_block": 2, "pooling": "concat(max, mean)",
            "head": [256, 128, num_classes], "head_dropout": [0.25, 0.5],
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "temporal_output_step_ms": 4000 / fs,
            "kernel_policy": "scale fs/100; round even kernels up to odd",
            "initialization": "Conv/Linear Kaiming normal fan_in, bias zero; BN weight one/bias zero",
            "comparison_limit": "frequency changes kernels, parameter count and temporal output step",
            "upstream_commit": "cdbf4e66d7e57d9b6a2657b6024716212b8d0afa",
        }

    def forward(self, signal):
        if signal.ndim != 3 or signal.shape[1] != 12:
            raise ValueError("Model input must have shape [batch, 12, samples]")
        features = self.features(signal)
        pooled = torch.cat((features.amax(dim=-1), features.mean(dim=-1)), dim=1)
        return self.head(pooled)


def build_model(fs: int, num_classes: int = 5) -> WangResNet1d:
    return WangResNet1d(fs, num_classes)
