"""Inkling-2 image processor: pad to a multiple of the 40x40 token patch (fill -1 on the
0-255 scale), rescale 1/255 and normalize (0.5, 0.5), then cut one patch per image token.

Transcribed from the transformers Inkling2ImageProcessor (no resize: the model reads
native resolution, 40..4096px recommended). The shared driver converts PIL images to RGB
before process(); the reference composites transparency onto a checkerboard instead --
transparent PNGs therefore differ on the background only.
"""

from __future__ import annotations

import math
import struct
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash

from freetoken.models.inkling2.config import IMAGE_TOKEN_ID


class Inkling2MMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = hf_config.vision_config
        self.image_token_id = int(getattr(hf_config, "image_token_id", IMAGE_TOKEN_ID))
        self.placeholder = [self.image_token_id]
        patch = int(getattr(vc, "native_patch_size", None) or vc.patch_size)
        merge = int(getattr(vc, "adapter_merge_size", None) or vc.spatial_merge_size)
        self.token_patch = patch * merge

    def process(self, images: list[Any]) -> list[MMItem]:
        import numpy as np

        tp = self.token_patch
        items: list[MMItem] = []
        for pil in images:
            arr = torch.from_numpy(np.asarray(pil, dtype=np.uint8)).permute(2, 0, 1).float()
            channels, height, width = arr.shape
            rows, cols = math.ceil(height / tp), math.ceil(width / tp)
            padded = torch.full((channels, rows * tp, cols * tp), -1.0)
            padded[:, :height, :width] = arr
            padded = (padded / 255.0 - 0.5) / 0.5
            patches = padded.reshape(channels, rows, tp, cols, tp).permute(1, 3, 2, 4, 0)
            pixel_values = patches.flatten(0, 1).to(torch.bfloat16)
            grid = [1, rows, cols]
            item_hash = content_hash(pixel_values, struct.pack("<3i", *grid))
            items.append(
                MMItem(
                    modality="image",
                    hash=item_hash,
                    pad_value=mm_pad_value(item_hash),
                    offsets=[],
                    feature=pixel_values,
                    model_specific_data={"grid_thw": grid},
                )
            )
        return items

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        t, rows, cols = item.grid_thw
        return PromptReplacement(full=[self.image_token_id] * (t * rows * cols))

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        tp = self.token_patch
        return [MMItem(
            modality="image",
            hash=0,
            pad_value=0,
            offsets=[[0, 1]],
            feature=torch.zeros(1, tp, tp, 3, dtype=dtype, device=device),
            model_specific_data={"grid_thw": [1, 1, 1]},
        )]


__all__ = ["Inkling2MMProcessor"]
