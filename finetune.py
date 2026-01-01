import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['HF_HOME'] = '/data1/yizhuo/Spatial-VLA/ckpt'
#os.environ['HUGGINGFACE_HUB_CACHE'] = '/data1/yizhuo/Spatial-VLA/ckpt/hub'
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import numpy as np
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from models.utils.eval_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from models.hf.configuration_prismatic import OpenVLAConfig
from models.hf.modeling_prismatic import OpenVLAForActionPrediction
from models.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from models.hf.load import load
from models.action_heads import L1RegressionActionHead, DiffusionActionHead, FullInjectActionHead
from models.projectors import ProprioProjector, NoisyActionProjector, AlignProjector
from models.backbones.llm import PurePromptBuilder
from models.backbones.spatial.vggt.models.vggt import VGGT
from models.backbones.spatial.vggt.utils.load_fn import preprocess_normed_images
from models.utils.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from models.utils.pooling_utils import custom_pooling
from models.utils.action_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from models.utils.train_utils import (
    compute_smoothened_metrics,
    count_parameters,
    log_metrics_to_wandb,
    remove_ddp_in_checkpoint,
    wrap_ddp
)
from dataset.utils.data_utils import PaddedCollatorForActionPrediction
from dataset.utils.action_tokenizer import ActionTokenizer
from dataset.utils.rlds.utils.data_utils import save_dataset_statistics
from dataset.utils.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    ACTION_TOKEN_BEGIN_IDX
)
from dataset import get_vla_dataset_and_collator

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # pretrain models
    model_type: str = "vlm"
    vla_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    vggt_path: str = "official_ckpts/vggt_model.pt"
    
    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_full_injection: bool = False
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    use_spatial: bool = False
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training 
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    vla_layers_align: int = -1                       # which layer of the VLA hidden state to align, 33 for total
    vggt_layers_align: int = -1                      # which layer of the VGGT hidden state to align, 24 for total
    align_loss_coeff: float = 0.5

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100000             # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200000                          # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    scheduler: str = 'MultiStepLR'                   # "MultiStepLR" or "CosineAnnealingLR"                                                
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!
    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print('loaded!!!!!!!!!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    align_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
) -> None:
   
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"
    
    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    dist.barrier()

    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)

        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")
        
        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )
        
        if cfg.use_spatial and align_projector is not None:
            torch.save(align_projector.state_dict(), checkpoint_dir / f"align_projector--{checkpoint_name_suffix}")

        if (cfg.use_l1_regression or cfg.use_diffusion or cfg.use_full_injection) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.model_type == "vlm":
            config = AutoConfig.from_pretrained("vla-qwen/configs/config.json")
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)  # Create a new model with configuration, the parameters are randomly initialized
            # print(new_state_dict['action_queries.weight'])

            #action-query
            new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
            missing_keys, unexpected_keys = base_vla.load_state_dict(new_state_dict, strict=False)
            
        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
                cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )


        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
        
        # Wait for merged model to be saved
        dist.barrier()


def run_forward_pass(
    phase,
    vla,
    action_head,
    spatial_model,
    noisy_action_projector,
    proprio_projector,
    align_projector,
    layers_align,
    align_loss_coeff,
    batch,
    action_tokenizer,
    processor,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_full_injection,
    use_proprio,
    use_film,
    use_spatial,
    num_patches,
    num_images_in_input,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
        )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    if use_spatial:
        # vla hidden states
        projected_embeddings = output.projector_features
        vision_length = projected_embeddings.shape[-2] - 1 if use_proprio else projected_embeddings.shape[-2]
        vla_hidden = output.hidden_states[layers_align[0]]  # 33 for total layers of VLA-7B
        boi_ids = 2
        vision_hidden = vla_hidden[:, boi_ids: vision_length+boi_ids, :].clone()  # [bs, 512, 4096]

        # VGGT hidden states
        spatial_model.eval()
        unnorm_imgs = preprocess_normed_images(
            batch['pixel_values'], processor.image_processor, num_images_in_input
        ).to(device_id)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            vggt_output = spatial_model(unnorm_imgs)
        agg_vggt_hidden = vggt_output["features"][layers_align[1]]  # 24 for total layers of VGGT
        patch_start_idx = vggt_output["patch_start_idx"]
        original_img = vggt_output["images"]
        vggt_hidden = agg_vggt_hidden[:, :, patch_start_idx:, :]

        # Resample VGGT hidden states to match the resolution of VLA hidden states
        H, W = original_img.shape[-2:]
        patch_h, patch_w = H // spatial_model.patch_size, W // spatial_model.patch_size
        vggt_hidden = custom_pooling(vggt_hidden, (patch_h, patch_w), (H, W), vision_hidden, "bilinear", True)

        # calculate align loss
        with torch.autocast("cuda", dtype=torch.bfloat16):
            align_loss = align_projector(vision_hidden, vggt_hidden)
    else:
        align_loss = torch.tensor(0.0, device=device_id)


    if not (use_l1_regression or use_diffusion or use_full_injection):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                "align_loss": align_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        if use_l1_regression:
            # Predict action
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )
        if use_full_injection:
            multi_layer_hidden_states = []

            for item in output.hidden_states:
                text_hidden_states = item[:, num_patches:-1]
                # Get hidden states for action portion of response
                batch_size = batch["input_ids"].shape[0]
                # actions_hidden_states = text_hidden_states[:, -1, :].reshape(batch_size, 1, -1).to(torch.bfloat16)
                actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, 1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
                task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches , -1)
                all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2) # oral feature + action feature
                multi_layer_hidden_states.append(all_hidden_states)
            multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)

            predicted_actions = action_head.module.predict_action(
                multi_layer_hidden_states,
                proprio=None,
                proprio_projector=None,
                phase=phase
                )

            loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "align_loss": align_loss.item(),
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = use_l1_regression or use_full_injection
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            if compute_diffusion_l1:
                print('curr: ',curr_action_l1_loss.item())
                # print('next: ',next_actions_l1_loss.item())
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss + align_loss * align_loss_coeff, metrics


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)

def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """ 

    global RAW_STATE_DICT

    # Trim trailing forward slash ('/') in VLA path if it exists
    if cfg.model_type == "vla":
        cfg.vla_path = cfg.vla_path.rstrip("/")
        print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}", mode="online")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
        f"\tACTION_TOKEN_BEGIN_IDX: {ACTION_TOKEN_BEGIN_IDX}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes

    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    if cfg.model_type == 'vlm':
        hf_token = ''
        vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        #vlm是一个承载器，load 可用的 checkpoint，这个参数最后还是要放回vla，所以最后forward是用的还是vla
        # config有点像定义这个model很多参数的文件
        config = AutoConfig.from_pretrained("vla-qwen/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict

    else:
        RAW_STATE_DICT ={}
        vla = OpenVLAForActionPrediction.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()

    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    if cfg.use_spatial:
        hf_model = VGGT.from_pretrained(cfg.vggt_path)

        vggt_model = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        )

        vggt_model.load_state_dict(hf_model.state_dict(), strict=False)
        vggt_model = vggt_model.to(device_id)

        align_projector = init_module(
            AlignProjector, "align_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim,
            "vggt_dim": vggt_model.embed_dim,
            "align_loss_type": "cosine",
            "use_vlm_norm": True,}
        )

        # vla.set_version(cfg.version)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    if cfg.use_full_injection:
        action_head = init_module(
        FullInjectActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim, 
            "hidden_dim": vla.module.llm_dim, 
            "action_dim": ACTION_DIM,
            "use_pro_version": True,
            },
        to_bf16=True,
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim, 
            "hidden_dim": vla.module.llm_dim, 
            "action_dim": ACTION_DIM
            },
        to_bf16=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input() # 512
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    # 取决于模型是否会将输入的 proprio 数据放入llm还是action head
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer，这里的参数都是要保存的参数
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion or cfg.use_full_injection:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    if cfg.use_spatial:
        trainable_params += [param for param in align_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]
    # Create learning rate scheduler
    if cfg.scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(
            optimizer,
            milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
            gamma=0.1,  # Multiplicative factor of learning rate decay
        )
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_steps,  # Total number of steps for the cosine annealing
            eta_min=cfg.learning_rate * 1e-3,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {cfg.scheduler}")

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1
    # Create training and optional validation datasets
    train_dataset, val_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.dataset_name,
        processor.image_processor.apply_transform,
        processor.tokenizer,
        PurePromptBuilder,
        tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        padding_side="right",
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        image_aug=cfg.image_aug,
    )
    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "align_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                phase="Training",
                vla=vla,
                action_head=action_head,
                spatial_model=vggt_model if cfg.use_spatial else None,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                align_projector = align_projector if cfg.use_spatial else None,
                layers_align=(cfg.vla_layers_align, cfg.vggt_layers_align) if cfg.use_spatial else None,
                align_loss_coeff=cfg.align_loss_coeff if cfg.use_spatial else 0.0,
                batch=batch,
                action_tokenizer=action_tokenizer,
                processor=processor if cfg.use_spatial else None,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_full_injection=cfg.use_full_injection,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                use_spatial=cfg.use_spatial,
                num_patches=NUM_PATCHES,
                num_images_in_input=cfg.num_images_in_input if cfg.use_spatial else None,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )
            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    align_projector = align_projector if cfg.use_spatial else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion or cfg.use_full_injection) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
