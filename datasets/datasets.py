"""
datasets.py

Dataset classes for the project.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from utils.action_tokenizer import ActionTokenizer
from models.backbones.llm import PromptBuilder

# @dataclass 自动生成 __init__ 方法    
@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: Any
    prompt_builder_fn: Type[PromptBuilder]