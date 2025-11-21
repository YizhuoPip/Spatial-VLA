"""
一次性得到做成dataloader的各个组件
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from models.backbones.llm import PromptBuilder
from models.backbones.vision import ImageTransform
from dataset.utils.data_utils import PaddedCollatorForActionPrediction
from dataset.utils.action_tokenizer import ActionTokenizer
from dataset.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    resize_resolution: Tuple[int, int],
    shuffle_buffer_size: int = 100_000,
    padding_side: str = "right",
    predict_stop_token: bool = True,
    use_wrist_image: bool = False,
    use_proprio: bool = False,
    image_aug: bool = False,
    episodic: bool = False,
) -> Tuple[Dataset, Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """
    Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions.

    Args:
        data_root_dir: Path to the directory containing the RLDS dataset.
        data_mix: Name of the dataset mix to use.
        image_transform: ImageTransform object to apply to the images.
        tokenizer: PreTrainedTokenizerBase object to tokenize the text.
        prompt_builder_fn: Type[PromptBuilder] object to build the prompt.
        resize_resolution: Tuple[int, int] containing the resolution to resize the images to.
        shuffle_buffer_size: int containing the size of the shuffle buffer.
        padding_side: str containing the side to pad the sequences to.
        predict_stop_token: bool indicating whether to predict the stop token.
        use_wrist_image: bool indicating whether to use the wrist image.
        use_proprio: bool indicating whether to use the proprioceptive features.
        image_aug: bool indicating whether to apply image augmentation.
        episodic: bool indicating whether to use episodic dataset.
    """
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token, use_wrist_image=use_wrist_image, use_proprio=use_proprio
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    train_dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
    )

    val_dataset = None

    return train_dataset, val_dataset, action_tokenizer, collator
