"""
materialize.py done

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
相当于是一个dict 从字典去对应这个project中预定义的model架构
"""

from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from models.backbones.llm import LLMBackbone, Qwen25LLMBackbone
from models.backbones.vision import (
    DinoSigLIPViTBackbone,
    ImageTransform,
    VisionBackbone,
)
from models.backbones.vlms import PrismaticVLM

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {
    "dinosiglip-vit-so-224px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinosiglip-vit-so-384px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
}


# === Language Model Registry ===
LLM_BACKBONES = {
    "qwen25-0_5b-pure": {"cls": Qwen25LLMBackbone, "kwargs": {}},
    "qwen25-0_5b-extra": {"cls": Qwen25LLMBackbone, "kwargs": {"num_extra_tokens": 256}},
    "qwen25-1_5b-pure": {"cls": Qwen25LLMBackbone, "kwargs": {}},
    "qwen25-3b-pure": {"cls": Qwen25LLMBackbone, "kwargs": {}},
}

# fmt: on


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            **llm_cfg["kwargs"],
        )
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")


def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
) -> PrismaticVLM:
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    return PrismaticVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
    )
