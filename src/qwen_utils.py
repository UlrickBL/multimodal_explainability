from __future__ import annotations
import requests
import base64
import math
import os
import sys
import warnings
from functools import lru_cache
from io import BytesIO

import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io as tv_io, transforms
from torchvision.transforms import InterpolationMode

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"Absolute aspect ratio must be < {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width):.1f}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar

def fetch_image(ele: dict, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    raw = ele.get("image") or ele.get("image_url")
    if isinstance(raw, Image.Image):
        image_obj = raw.convert("RGB")
    elif isinstance(raw, str):
        if raw.startswith(("http://", "https://")):
            image_obj = Image.open(requests.get(raw, stream=True).raw).convert("RGB")
        elif raw.startswith("file://"):
            image_obj = Image.open(raw[7:]).convert("RGB")
        elif raw.startswith("data:image"):
            _, b64 = raw.split("base64,", 1)
            image_obj = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        else:
            image_obj = Image.open(raw).convert("RGB")
    else:
        raise ValueError(f"Unrecognised image spec: {raw!r}")

    if "resized_height" in ele and "resized_width" in ele:
        rh, rw = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=size_factor
        )
    else:
        w, h = image_obj.size
        min_px = ele.get("min_pixels", MIN_PIXELS)
        max_px = ele.get("max_pixels", MAX_PIXELS)
        rh, rw = smart_resize(h, w, factor=size_factor, min_pixels=min_px, max_pixels=max_px)

    return image_obj.resize((rw, rh))


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos: list[dict] = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conv in conversations:
        for msg in conv:
            if isinstance(msg["content"], list):
                for ele in msg["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, None]:
    vision_infos = extract_vision_info(conversations)
    image_inputs: list[Image.Image] = []
    for info in vision_infos:
        if "image" in info or "image_url" in info or info.get("type") in ("image", "image_url"):
            image_inputs.append(fetch_image(info))
    return (image_inputs if image_inputs else None), None
