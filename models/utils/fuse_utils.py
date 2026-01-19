import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFusion(nn.Module):
    """
    Conditional fusion module using cross-attention with self-gating.
    Q: LLM visual tokens (作为condition和query)
    K, V: spatial tokens
    使用Q本身来生成gate，控制spatial feature的融合
    """

    def __init__(
        self,
        dim_q,
        dim_kv,
        dim_out=None,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        use_ln=True,
        use_self_gate=False,
        gate_type="token-wise",
    ):
        super().__init__()

        dim_out = dim_out or dim_q
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        assert (
            dim_out % num_heads == 0
        ), "dim_out must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5
        self.use_self_gate = use_self_gate
        self.gate_type = gate_type

        # optional layer norm
        self.norm_q = nn.LayerNorm(dim_q) if use_ln else nn.Identity()
        self.norm_kv = nn.LayerNorm(dim_kv) if use_ln else nn.Identity()

        # projections
        self.q_proj = nn.Linear(dim_q, dim_out)
        self.k_proj = nn.Linear(dim_kv, dim_out)
        self.v_proj = nn.Linear(dim_kv, dim_out)

        # output projection
        self.out_proj = nn.Linear(dim_out, dim_out)

        # Self-gating mechanism using q_tokens
        if self.use_self_gate:
            if gate_type == "token-wise":
                # Token-wise gate: generate gate for each query token
                self.gate_proj = nn.Linear(dim_q, dim_out)
                self.gate_activation = nn.Sigmoid()
            elif gate_type == "global":
                # Global gate: single gate value for all tokens
                self.gate_proj = nn.Sequential(
                    nn.Linear(dim_q, dim_out // 4),
                    nn.ReLU(),
                    nn.Linear(dim_out // 4, 1),
                    nn.Sigmoid()
                )
            elif gate_type == "attention":
                # Attention-based gate: use q_tokens to attend to output features
                self.gate_attn_proj = nn.Linear(dim_q, dim_out)
                self.gate_attn_drop = nn.Dropout(attn_drop)
            else:
                raise ValueError(f"Unsupported gate_type: {gate_type}")

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_tokens, kv_tokens, attn_mask=None):
        """
        Args:
            q_tokens: (B, Nq, Cq) - LLM visual tokens (作为condition和query)
            kv_tokens: (B, Nk, Ck) - spatial tokens  
            attn_mask: optional (B, Nq, Nk) or broadcastable
        Returns:
            fused_tokens: (B, Nq, dim_out)
        """
        B, Nq, _ = q_tokens.shape
        Nk = kv_tokens.shape[1]

        # norm
        q_tokens_norm = self.norm_q(q_tokens)
        kv_tokens_norm = self.norm_kv(kv_tokens)

        # linear projections
        q = self.q_proj(q_tokens_norm)  # (B, Nq, D)
        k = self.k_proj(kv_tokens_norm)  # (B, Nk, D)
        v = self.v_proj(kv_tokens_norm)  # (B, Nk, D)

        # reshape for multi-head
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # attention: (B, H, Nq, Nk)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # weighted sum
        out = attn @ v  # (B, H, Nq, head_dim)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)

        # Apply self-gating if enabled
        if self.use_self_gate:
            gate = self._compute_gate(q_tokens_norm, out, B, Nq)
            out = out * gate

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

    def _compute_gate(self, q_tokens, out, B, Nq):
        """
        使用q_tokens本身来生成gate，控制输出特征
        Args:
            q_tokens: (B, Nq, Cq) - normalized query tokens (LLM视觉tokens)
            out: (B, Nq, dim_out) - 当前融合后的特征
            B: batch size
            Nq: query token数量
        Returns:
            gate: (B, Nq, dim_out) 或 (B, Nq, 1) - gate weights
        """
        if self.gate_type == "token-wise":
            # 直接使用q_tokens生成每个token的gate
            gate = self.gate_proj(q_tokens)  # (B, Nq, dim_out)
            gate = self.gate_activation(gate)  # (B, Nq, dim_out)
            
        elif self.gate_type == "global":
            # 平均池化q_tokens，生成全局gate
            q_pooled = q_tokens.mean(dim=1)  # (B, Cq)
            gate = self.gate_proj(q_pooled)  # (B, 1)
            gate = gate.unsqueeze(1).expand(B, Nq, -1)  # (B, Nq, 1)
            
        elif self.gate_type == "attention":
            # 使用注意力机制生成gate：q_tokens作为query，out作为key和value
            gate_q = self.gate_attn_proj(q_tokens)  # (B, Nq, dim_out)
            gate_q = gate_q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, head_dim)
            
            out_reshaped = out.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, head_dim)
            
            # 计算注意力权重（自注意力）
            gate_attn = (gate_q @ out_reshaped.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nq)
            gate_attn = F.softmax(gate_attn, dim=-1)  # 在target维度上softmax
            gate_attn = self.gate_attn_drop(gate_attn)
            
            # 使用注意力权重聚合信息
            gate = (gate_attn @ out_reshaped)  # (B, H, Nq, head_dim)
            gate = gate.transpose(1, 2).contiguous().view(B, Nq, -1)  # (B, Nq, dim_out)
            gate = torch.sigmoid(gate)  # (B, Nq, dim_out)
            
        else:
            raise ValueError(f"Unsupported gate_type: {self.gate_type}")
            
        return gate
