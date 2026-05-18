from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from typing_extensions import Unpack

from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPMLP, eager_attention_forward, CLIPPreTrainedModel, CLIPVisionEmbeddings
from transformers.utils import TransformersKwargs, auto_docstring


class CLIPSurgeryAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CLIPVisionConfig | CLIPTextConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output_ori, attn_weights_ori = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )
        
        # Dual path, replace q & k by v
        keys = values
        queries = keys 
        
        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )
        
        attn_output_ori = attn_output_ori.reshape(batch_size, seq_length, -1).contiguous()
        attn_output_ori = self.out_proj(attn_output_ori)
        
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return [(attn_output, attn_weights), (attn_output_ori, attn_weights_ori)]


def _feature_take_indices(
    num_features: int, indices: Optional[Union[int, List[int]]] = None
) -> Tuple[List[int], int]:
    """Which encoder layer indices to record (same convention as timm). ``None`` → all layers."""
    if indices is None:
        indices = num_features
    if isinstance(indices, int):
        assert 0 < indices <= num_features
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            assert 0 <= idx < num_features, f"layer index {idx} out of range [0, {num_features})"
            take_indices.append(idx)
    return take_indices, max(take_indices) if take_indices else 0


def _merge_surgery_dual_path(hidden_states: Union[torch.Tensor, list]) -> torch.Tensor:
    """CLS from original path, patch tokens from surgery path (NLD)."""
    if isinstance(hidden_states, list) and len(hidden_states) == 2:
        h, h_ori = hidden_states
        out = h.clone()
        out[:, 0, :] = h_ori[:, 0, :]
        return out
    return hidden_states


class CLIPSurgeryEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: CLIPVisionConfig | CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPSurgeryAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        if isinstance(self.self_attn, CLIPSurgeryAttention):
            if isinstance(hidden_states, list):
                residual_surgery, hidden_states_ori = hidden_states
                residual_ori = hidden_states_ori
                hidden_states_ori = self.layer_norm1(hidden_states_ori)
                outputs, outputs_ori = self.self_attn(
                    hidden_states=hidden_states_ori,
                    attention_mask=attention_mask,
                    **kwargs,
                )
                hidden_states, _ = outputs
                hidden_states_ori, _ = outputs_ori
                hidden_states_ori = residual_ori + hidden_states_ori
                residual_ori = hidden_states_ori
                hidden_states_ori = self.layer_norm2(hidden_states_ori)
                hidden_states_ori = self.mlp(hidden_states_ori)
                hidden_states_ori = residual_ori + hidden_states_ori
                hidden_states = residual_surgery + hidden_states  # skip ffn for the new path
                return [hidden_states, hidden_states_ori]
                
            # start of dual path
            else:
                residual = hidden_states
                hidden_states = self.layer_norm1(hidden_states)
                outputs, outputs_ori = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    **kwargs,
                )
                hidden_states, _ = outputs
                hidden_states_ori, _ = outputs_ori
                hidden_states_ori = residual + hidden_states_ori
                residual_ori = hidden_states_ori
                hidden_states_ori = self.layer_norm2(hidden_states_ori)
                hidden_states_ori = self.mlp(hidden_states_ori)
                hidden_states_ori = residual_ori + hidden_states_ori
                hidden_states = hidden_states + residual# skip ffn for the new path
                return [hidden_states, hidden_states_ori]
            
        else:
            # default behavior
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states
    
class CLIPSurgeryEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        layers = []
        depth = config.num_hidden_layers
        for i in range(depth):
            # apply architecture surgery on the last 6 blocks
            start_index = max(0, depth - 6)
            apply_surgery = i >= start_index
            attn_layer_i = CLIPSurgeryEncoderLayer if apply_surgery else CLIPEncoderLayer
            layers.append(attn_layer_i(config))
        self.layers = nn.ModuleList(layers)
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
        """
        hidden_states = inputs_embeds
        
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )
        # hidden_states is a list [hidden_states, hidden_states_ori] because last blocks use CLIPSurgeryAttention
        if isinstance(hidden_states, list) and len(hidden_states) == 2:
            hidden_states, hidden_states_ori = hidden_states
            hidden_states[:, 0, :] = hidden_states_ori[:, 0, :] # cls token from the original path, img tokens from the new path
            # in CLIP, BaseModelOutputWithPooling is returned after this forward pass
            #    BaseModelOutputWithPooling(
                #    last_hidden_state=last_hidden_state,
                #    pooler_output=pooled_output,
                #)
            # where pooled_output is the cls  token -> pooled_output = last_hidden_state[:, 0, :]
            # so since CLIPSurgery leaves cls token unchanged the output of GeoCLIP with surgery is same as original GeoCLIP
            # we only modify the hidden states with image patch tokens for visualization
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )

    def forward_intermediates(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        indices: Optional[Union[int, List[int]]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[BaseModelOutput, List[torch.Tensor]]:
        """Single pass through layers; returns final encoder output + list of merged NLD states (pre merge-at-end)."""
        take_indices, _ = _feature_take_indices(len(self.layers), indices)
        take_set = set(take_indices)
        hidden_states = inputs_embeds
        intermediates: List[torch.Tensor] = []

        for i, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )
            if i in take_set:
                intermediates.append(_merge_surgery_dual_path(hidden_states).clone())

        if isinstance(hidden_states, list) and len(hidden_states) == 2:
            hidden_states, hidden_states_ori = hidden_states
            hidden_states[:, 0, :] = hidden_states_ori[:, 0, :]

        return BaseModelOutput(last_hidden_state=hidden_states), intermediates


class CLIPSurgeryVisionTransformer(CLIPPreTrainedModel):
    config: CLIPVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPSurgeryEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.post_init()

    def forward_intermediates(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        indices: Optional[Union[int, List[int]]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[BaseModelOutputWithPooling, List[torch.Tensor]]:
        """One vision forward: same first return as :meth:`forward` + per-layer encoder hiddens (pre ``post_layernorm``).

        Returns
        -------
        vision_output : BaseModelOutputWithPooling
            Same fields as ``forward``: ``last_hidden_state`` (encoder merge, pre ``post_layernorm``),
            ``pooler_output`` (``post_layernorm`` on CLS only).
        layer_hiddens : list[Tensor]
            Merged encoder outputs before ``post_layernorm``, one entry per recorded layer.
        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs, intermediates = self.encoder.forward_intermediates(
            inputs_embeds=hidden_states,
            indices=indices,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return (
            BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
            ),
            intermediates,
        )

    def get_intermediate_layers(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool | None = False,
        n: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> List[torch.Tensor]:
        """Pre–post_layernorm merged hiddens; ``n=None`` → all layers."""
        _, inter = self.forward_intermediates(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            indices=n,
            **kwargs,
        )
        return inter

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :] #pooled_output is the cls token
        #pooled_output = last_hidden_state # for visualization, return all tokens, no pooling
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )