import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.distributions.uniform import Uniform
import math
from typing import Dict, Optional, Tuple, Any
import logging

from utils.hub_mixin import CompatiblePyTorchModelHubMixin
from models.hrdt.model import HRDT


class ReasoningHead(nn.Module):
    """
    Auxiliary reasoning prediction head.
    Training-only scaffold — never saved, never used at inference.
    """
    def __init__(self, hidden_size=2176, vocab_size=32100, max_seq_len=150):
        super().__init__()
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, state_hidden, seq_len):
        positions = torch.arange(seq_len, device=state_hidden.device)
        pos_emb = self.pos_emb(positions)
        context = self.context_proj(state_hidden).unsqueeze(1) + pos_emb.unsqueeze(0)
        logits = self.output_proj(context)
        return logits


class LSAHead(nn.Module):
    """
    Latent Semantic Alignment head.
    Projects image-grounded action tokens to T5 reasoning embedding space.
    Training-only scaffold — never saved, never used at inference.
    """
    def __init__(self, hidden_size=2176, t5_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, t5_dim)
        )

    def forward(self, action_hidden, per_token=False):
        # action_hidden: (B, 16, hidden_size) — action tokens after img cross-attn
        if per_token:
            return self.proj(action_hidden)  # (B, 16, t5_dim) — project each token
        pooled = action_hidden.mean(dim=1)  # (B, hidden_size)
        return self.proj(pooled)            # (B, t5_dim)


class SigmoidTimestepSampler:
    """
    LogitNormal sampler  
    Sampling: u ~ N(mean, std), t = sigmoid(u)
    """
    def __init__(self, timestep_max=1.0, mean=0.0, std=1.0):
        self.timestep_max = timestep_max
        self.mean = mean  # Normal distribution mean
        self.std = std    # Normal distribution standard deviation
    
    def sample(self, shape):
        """
        LogitNormal sampling, which is sigmoid(randn(m,s))
        
        1. u ~ N(mean, std)
        2. t = sigmoid(u) 
        """
        # Generate normal distribution random numbers u ~ N(mean, std)
        u = torch.normal(mean=self.mean, std=self.std, size=shape)
        # Apply sigmoid transformation to get timesteps in (0,1) range
        t = torch.sigmoid(u)
        # Scale to [0, timestep_max]
        return t * self.timestep_max
    
    def visualize_distribution(self, num_samples=10000):
        """
        Visualize sampling distribution
        """
        samples = self.sample((num_samples,))
        return {
            'samples': samples,
            'mean': samples.mean().item(),
            'std': samples.std().item(),
            'min': samples.min().item(),
            'max': samples.max().item(),
            'config': f'LogitNormal(mean={self.mean}, std={self.std})'
        }


class ActionEncoder(nn.Module):
    """Action encoder that combines state and action adaptors"""
    
    def __init__(self, state_dim, action_dim, hidden_size, config):
        super().__init__()
        self.state_adaptor = self.build_condition_adapter(
            config['st_adaptor'],
            in_features=state_dim,
            out_features=hidden_size
        )
        self.action_adaptor = self.build_condition_adapter(
            config['act_adaptor'],
            in_features=action_dim,
            out_features=hidden_size
        )
    
    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def encode_state(self, state_tokens):
        return self.state_adaptor(state_tokens)
    
    def encode_action(self, action_tokens):
        return self.action_adaptor(action_tokens)


class HRDTRunner(
        nn.Module,
        CompatiblePyTorchModelHubMixin,
        repo_url="https://huggingface.co/hongzhe2002/H-RDT/"
    ):
    def __init__(self, *, state_dim, action_dim,
                 pred_horizon, config, act_pos_emb_config=None, img_pos_emb_config=None, lang_pos_emb_config=None,
                 max_img_len=None, max_lang_len=None,
                 training_mode='lang',
                 mode='pretrain',
                 pretrained_backbone_path=None,
                 dtype=torch.bfloat16):
        super(HRDTRunner, self).__init__()
        # Create diffusion model
        hidden_size = config['hrdt']['hidden_size']
        self.gradient_checkpointing = False
        self.hidden_size = hidden_size
        self.training_mode = training_mode
        self.mode = mode  # 'pretrain' or 'finetune'
        
        # Validate mode
        if mode not in ['pretrain', 'finetune']:
            raise ValueError(f"mode must be 'pretrain' or 'finetune', got {mode}")

        # Create H-RDT model
        self.model = HRDT(
            horizon=pred_horizon,
            config=config['hrdt'],
            x_pos_emb_config=act_pos_emb_config,
            img_pos_emb_config=img_pos_emb_config,
            lang_pos_emb_config=lang_pos_emb_config,
            max_img_len=max_img_len,
            max_lang_len=max_lang_len,
            training_mode=training_mode,
            dtype=dtype,
        )

        # Image features adapter - use dimensions from config
        self.img_adapter = self.build_condition_adapter(
            config.get('img_adapter', 'mlp2x_silu'),
            in_features=config.get('vision', {}).get('feature_dim', 2048),  # Default to ResNet50 dim
            out_features=hidden_size
        )
        
        # Action encoder (state and action adaptors)
        self.action_encoder = ActionEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            config=config
        )

        # Language features adapter - use dimensions from config
        self.lang_adapter = self.build_condition_adapter(
            config.get('lang_adapter', 'mlp2x_silu'),
            in_features=config.get('text', {}).get('feature_dim', 768),  # Default to DistilBERT dim
            out_features=hidden_size
        )

        # Create noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.timestep_max = noise_scheduler_config['timestep_max']
        
        sampler_type = noise_scheduler_config.get('sampler_type', 'sigmoid')
        if sampler_type == 'uniform':
            self.timestep_sampler = Uniform(0, self.timestep_max)
        elif sampler_type == 'sigmoid':
            mean = noise_scheduler_config.get('sigmoid_mean', 0.0)
            std = noise_scheduler_config.get('sigmoid_std', 1.0)
            self.timestep_sampler = SigmoidTimestepSampler(self.timestep_max, mean, std)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # TimeNoise config
        self.time_noise_a = config["time_noise"]["a"]
        self.time_noise_beta_m = config["time_noise"]["beta_m"]
        
        self.img_pos_emb_config = img_pos_emb_config

        # Load pretrained backbone weights if in finetune mode
        if mode == 'finetune' and pretrained_backbone_path is not None:
            self.load_pretrained_backbone(pretrained_backbone_path)

        # Print model size
        print("Model params: %e" % sum(p.numel() for p in self.parameters()))

    def load_pretrained_backbone(self, pretrained_path):
        """Load pretrained backbone weights while keeping action encoder and decoder fresh"""
        logging.info(f"Loading pretrained backbone from {pretrained_path}")
        
        # Load checkpoint
        if pretrained_path.endswith('.safetensors'):
            import safetensors.torch
            checkpoint = safetensors.torch.load_file(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter out action encoder and action decoder weights
        backbone_state_dict = {}
        action_related_keys = []
        
        for key, value in state_dict.items():
            # Skip action encoder (state_adaptor, act_adaptor) and action decoder weights
            if any(pattern in key for pattern in [
                'st_adaptor', 'act_adaptor', 'action_encoder', 
                'final_layer', 'action_decoder'
            ]):
                action_related_keys.append(key)
                continue

            # Skip img_pos_emb if there is a shape mismatch
            if key == 'model.img_pos_emb':
                current_shape = self.model.img_pos_emb.shape
                if value.shape != current_shape:
                    logging.info(f"Skipping img_pos_emb due to shape mismatch: {value.shape} vs {current_shape}")
                    continue

            backbone_state_dict[key] = value
        
        # Load backbone weights
        missing_keys, unexpected_keys = self.load_state_dict(backbone_state_dict, strict=False)
        
        logging.info(f"Loaded backbone with {len(backbone_state_dict)} parameters")
        logging.info(f"Skipped action-related keys: {action_related_keys}")
        logging.info(f"Missing keys: {missing_keys}")
        logging.info(f"Unexpected keys: {unexpected_keys}")
        logging.info("Action encoder and decoder initialized from scratch for finetune mode")

    @classmethod
    def from_pretrained_for_finetune(cls, pretrained_path, state_dim, action_dim, pred_horizon, config, **kwargs):
        """Create model in finetune mode with pretrained backbone"""
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            config=config,
            mode='finetune',
            pretrained_backbone_path=pretrained_path,
            **kwargs
        )

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def gradient_checkpointing_enable(self, value=True):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = value
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(value)

    def compute_loss(self, state_tokens=None, action_gt=None, image_tokens=None, lang_tokens=None, lang_attn_mask=None, reasoning_token_ids=None, reasoning_head=None, reasoning_lambda=0.1, lsa_head=None, lsa_embeddings=None, lsa_attn_mask=None, lsa_lambda=0.1,
                     dense_lsa_targets=None, dense_lsa_mask=None, use_dense_lsa=False,
                     dense_lsa_targets_R=None, dense_lsa_mask_R=None):
        """
            img_tokens: (batch_size, img_len, img_token_dim)
            state_tokens: (batch_size, chunk_size, action_dim), 
            action_gt: (batch_size, chunk_size, action_dim), ground-truth actions for supervision
            lang_tokens: (batch_size, L, hidden_size), language features (unpooled)
            lang_attn_mask: (batch_size, L), attention mask for language tokens
        """
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        dtype = image_tokens.dtype

        noise = torch.randn(action_gt.shape, dtype=dtype, device=device)
        timesteps = self.timestep_sampler.sample((batch_size,)).to(device)
        
        broadcasted = timesteps.view(-1, 1, 1)
        noisy_action = (action_gt * broadcasted + noise * (1 - broadcasted)).to(dtype=dtype)

        img_c = self.img_adapter(image_tokens)

        # Process language features - handle None case
        lang_c = None
        if lang_tokens is not None:
            lang_c = self.lang_adapter(lang_tokens)  # [B, L, D] - keep unpooled for cross attention

        # state/action using action encoder
        state_traj = self.action_encoder.encode_state(state_tokens)
        action_traj = self.action_encoder.encode_action(noisy_action)
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)

        use_reasoning = (reasoning_token_ids is not None and reasoning_head is not None)
        use_lsa = (lsa_head is not None and lsa_embeddings is not None and not use_dense_lsa)
        use_dense = (use_dense_lsa and lsa_head is not None and dense_lsa_targets is not None)
        need_hidden = use_reasoning or use_lsa or use_dense

        if need_hidden:
            pred, hidden = self.model(state_action_traj, timesteps, img_c=img_c, lang_c=lang_c,
                                      lang_attn_mask=lang_attn_mask, return_hidden=True)
        else:
            pred = self.model(state_action_traj, timesteps, img_c=img_c, lang_c=lang_c,
                              lang_attn_mask=lang_attn_mask)

        target = action_gt - noise
        diff_loss = F.mse_loss(pred, target)

        # Reasoning auxiliary loss (token prediction)
        if use_reasoning:
            state_hidden = hidden[:, 0, :]
            seq_len = reasoning_token_ids.shape[1]
            reasoning_logits = reasoning_head(state_hidden, seq_len)
            reasoning_loss = F.cross_entropy(
                reasoning_logits.reshape(-1, reasoning_logits.shape[-1]),
                reasoning_token_ids.reshape(-1).long(),
                ignore_index=0
            )
            total_loss = diff_loss + reasoning_lambda * reasoning_loss
            return {"diff_loss": diff_loss, "reasoning_loss": reasoning_loss, "loss": total_loss}

        # Dense LSA loss (Option D: per-token phase alignment)
        if use_dense:
            action_hidden = hidden[:, 1:, :]                  # (B, K, hidden_size)
            projected = lsa_head(action_hidden, per_token=True)  # (B, K, t5_dim) ONE projection
            p_norm = F.normalize(projected, dim=-1)
            # --- TRACK 1 (single active hand for single-arm; hand-1 for bimanual) ---
            r_norm = F.normalize(dense_lsa_targets.float(), dim=-1)  # (B, K, t5_dim)
            cos = (p_norm * r_norm).sum(dim=-1)               # (B, K)
            loss_tok = (1 - cos)
            if dense_lsa_mask is not None:
                m = dense_lsa_mask.float()
                lsa_track1 = (loss_tok * m).sum() / m.sum().clamp(min=1)
            else:
                lsa_track1 = loss_tok.mean()
            # --- TRACK 2 (empty/0 for single-arm; hand-2 for bimanual) ---
            lsa_track2 = projected.new_zeros(())              # scalar 0 on correct device/dtype
            if dense_lsa_targets_R is not None and dense_lsa_mask_R is not None:
                r_norm_R = F.normalize(dense_lsa_targets_R.float(), dim=-1)
                cos_R = (p_norm * r_norm_R).sum(dim=-1)
                loss_tok_R = (1 - cos_R)
                m_R = dense_lsa_mask_R.float()
                # when m_R is all-zero (single-arm), this is 0/clamp(1) = 0 -> reduces to current
                lsa_track2 = (loss_tok_R * m_R).sum() / m_R.sum().clamp(min=1)
            lsa_loss_val = lsa_track1 + lsa_track2
            total_loss = diff_loss + lsa_lambda * lsa_loss_val
            return {"diff_loss": diff_loss, "lsa_loss": lsa_loss_val, "loss": total_loss}

        # LSA loss (latent semantic alignment)
        if use_lsa:
            action_hidden = hidden[:, 1:, :]  # (B, 16, hidden_size) image-grounded
            projected = lsa_head(action_hidden)  # (B, t5_dim)
            if lsa_attn_mask is not None:
                mask = lsa_attn_mask.unsqueeze(-1).float()
                reasoning_mean = (lsa_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                reasoning_mean = lsa_embeddings.mean(dim=1)
            p_norm = F.normalize(projected, dim=-1)
            r_norm = F.normalize(reasoning_mean.float(), dim=-1)
            lsa_loss_val = (1 - (p_norm * r_norm).sum(dim=-1)).mean()
            total_loss = diff_loss + lsa_lambda * lsa_loss_val
            return {"diff_loss": diff_loss, "lsa_loss": lsa_loss_val, "loss": total_loss}

        return {"diff_loss": diff_loss, "loss": diff_loss}

    @torch.no_grad()
    def predict_action(self, state_tokens=None, image_tokens=None, lang_tokens=None, lang_attn_mask=None):
        '''
        state_tokens: (batch_size, chunk_size, action_dim)
        image_tokens: (batch_size, img_len, in_feat_dim)
        lang_tokens (torch.Tensor): language features [B, L, hidden_size] (unpooled)
        lang_attn_mask: (batch_size, L), attention mask for language tokens
        
        return: (batch_size, chunk_size, action_dim), predicted action sequence
        '''
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        dtype = image_tokens.dtype

        img_c = self.img_adapter(image_tokens)

        # Process language features - handle None case
        lang_c = None
        if lang_tokens is not None:
            lang_c = self.lang_adapter(lang_tokens)  # [B, L, D] - keep unpooled for cross attention

        state_traj = self.action_encoder.encode_state(state_tokens)
        noisy_action = torch.randn((batch_size, self.pred_horizon, self.action_dim), dtype=dtype, device=device)
        timestep = torch.tensor([0.0], dtype=dtype, device=device)
        step_size = 1.0 / self.num_inference_timesteps

        for _ in range(self.num_inference_timesteps):
            action_traj = self.action_encoder.encode_action(noisy_action)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            pred = self.model(state_action_traj, timestep, img_c=img_c, lang_c=lang_c, lang_attn_mask=lang_attn_mask)
            noisy_action = pred * step_size + noisy_action
            timestep = timestep + step_size

        return noisy_action

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)