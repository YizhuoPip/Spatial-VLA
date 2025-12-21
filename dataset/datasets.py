"""
datasets.py

Dataset classes for the project.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from dataset.utils.oxe import get_oxe_dataset_kwargs_and_weights, OXE_NAMED_MIXTURES
from dataset.utils.rlds import make_interleaved_dataset, make_single_dataset
from dataset.utils.action_tokenizer import ActionTokenizer
from dataset.utils.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, IGNORE_INDEX, NUM_ACTIONS_CHUNK, NUM_TOKENS
from models.backbones.llm import PromptBuilder, QwenPromptBuilder
from models.backbones.vision import ImageTransform

# @dataclass 自动生成 __init__ 方法    
@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an RLDS batch to a model input batch.

        :param rlds_batch: The RLDS batch to convert.
        """
        dataset_name = rlds_batch["dataset_name"]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]
        
        #get action 
        current_action = rlds_batch["action"][0]
        future_actions = rlds_batch["action"][1:]

        if isinstance(self.action_tokenizer.tokenizer, Qwen2TokenizerFast):
            self.prompt_builder_fn = QwenPromptBuilder
            prompt_builder = self.prompt_builder_fn("openvla")

            future_actions_string = self.action_tokenizer(future_actions)
            current_action_string = self.action_tokenizer(current_action)
            action_chunk_string = [current_action_string] + future_actions_string
            flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]
            action_chunk_len = len(flattened_action_chunk_string) 

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ''},
            ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

            if len(input_ids) >= 3:
                del input_ids[-3] 
                del input_ids[-2] 
                del input_ids[-1] 

            if NUM_TOKENS<len(flattened_action_chunk_string):
                input_ids = input_ids + flattened_action_chunk_string[:NUM_TOKENS]
            else:
                remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
                extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
                
                input_ids = input_ids + flattened_action_chunk_string + extended_array
            labels = list(input_ids)
            action_chunk_len = NUM_TOKENS
        else:
            prompt_builder = self.prompt_builder_fn("openvla")
            current_action_string = self.action_tokenizer(current_action)
            future_action_string = ''.join(self.action_tokenizer(future_actions))
            #decode 之后的action token，然后与文本一起去encode
            action_chunk_string = current_action_string + future_action_string
            action_chunk_len = len(action_chunk_string)

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            
            # Tokenize
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # Image Transform
        pixel_values = self.image_transform(img)

        # `IGNORE_INDEX` 是一个特殊值,计算损失时忽略这个位置.
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # wrrist image
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        # Proprioception
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio
        return return_dict

class RLDSDataset(IterableDataset):
    def __init__(
            self,
            data_root_dir: Path,
            data_mix: str,
            batch_transform: RLDSBatchTransform,
            resize_resolution: Tuple[int, int],
            shuffle_buffer_size: int = 256_000,
            train: bool = True,
            image_aug: bool = False,
    ) -> None:
        """
        Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders.

        :param data_root_dir: Path to directory containing RLDS TFDS files
        :param data_mix: Name of the mix to use
        :param batch_transforme: RLDSBatchTransform to apply to each batch
        :param resize_resolution: Resolution to resize images to
        :param shuffle_buffer_size: Size of shuffle buffer
        """

        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")


        per_dataset_kwargs, per_dataset_weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=per_dataset_weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)
    
    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out
