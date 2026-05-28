import numpy as np
import timm
import torch
from torch import nn
from torch.nn import Identity
from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights

from location_encoders.satclip.location_encoder import (
    LocationEncoder,
    get_neural_network,
    get_positional_encoding,
)

from .surgery_vision_transformer import VisionTransformer


class ModifiedResNet(nn.Module):
    def __init__(
        self, layers, output_dim, heads, input_resolution=224, width=64, in_channels=3
    ):
        super().__init__()
        raise NotImplementedError(
            "ModifiedResNet path is not implemented for SatCLIP surgery in this repo."
        )


class SatCLIPSurgery(nn.Module):
    """SatCLIP model with CLIP-Surgery vision encoder."""

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: tuple[int, int, int, int] | int | str,
        vision_width: int,
        vision_patch_size: int,
        in_channels: int,
        # location
        le_type: str,
        pe_type: str,
        frequency_num: int,
        max_radius: int,
        min_radius: int,
        harmonics_calculation: str,
        legendre_polys: int = 10,
        sh_embedding_dims: int = 16,
        ffn: bool = True,
        num_hidden_layers: int = 2,
        capacity: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            # This branch is included for completeness but not expected to be used
            print("using modified resnet (surgery)")
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                in_channels=in_channels,
            )

        elif vision_layers == "moco_resnet18":
            print("using pretrained moco resnet18 (surgery)")
            weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.visual = timm.create_model(
                "resnet18", in_chans=in_chans, num_classes=embed_dim
            )
            self.visual.load_state_dict(
                weights.get_state_dict(progress=True), strict=False
            )
            self.visual.requires_grad_(False)
            self.visual.fc.requires_grad_(True)

        elif vision_layers == "moco_resnet50":
            print("using pretrained moco resnet50 (surgery)")
            weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.visual = timm.create_model(
                "resnet50", in_chans=in_chans, num_classes=embed_dim
            )
            self.visual.load_state_dict(
                weights.get_state_dict(progress=True), strict=False
            )
            self.visual.requires_grad_(False)
            self.visual.fc.requires_grad_(True)

        elif vision_layers == "moco_vit16":
            print("using pretrained moco vit16 (surgery)")
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]

            self.visual = VisionTransformer(
                in_chans=in_chans,
                num_classes=embed_dim,
                embed_dim=384,
                depth=12,
                num_heads=6,
            )
            self.visual.load_state_dict(
                weights.get_state_dict(progress=True), strict=False
            )
            self.visual.requires_grad_(False)
            self.visual.head.requires_grad_(True)
            self.visual.attn_pool = Identity()  # no pooling
            self.visual.num_prefix_tokens = 0  # return cls token too

        else:
            print("using surgery vision transformer")
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                img_size=image_resolution,
                patch_size=vision_patch_size,
                in_chans=in_channels,
                num_classes=embed_dim,
                embed_dim=vision_width,
                depth=vision_layers if isinstance(vision_layers, int) else 12,
                num_heads=vision_heads,
            )

        self.posenc = get_positional_encoding(
            name=le_type,
            harmonics_calculation=harmonics_calculation,
            legendre_polys=legendre_polys,
            min_radius=min_radius,
            max_radius=max_radius,
            frequency_num=frequency_num,
        ).double()
        self.nnet = get_neural_network(
            name=pe_type,
            input_dim=self.posenc.embedding_dim,
            num_classes=embed_dim,
            dim_hidden=capacity,
            num_layers=num_hidden_layers,
        ).double()
        self.location = LocationEncoder(
            self.posenc,
            self.nnet,
        ).double()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if getattr(self.visual, "attnpool", None) is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        if isinstance(
            self.visual,
            (timm.models.vision_transformer.VisionTransformer, VisionTransformer),
        ):
            return self.visual.patch_embed.proj.weight.dtype
        else:
            return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_location(self, coords):
        return self.location(coords.double())

    def forward(self, image, coords):
        image_features = self.encode_image(image)
        location_features = self.encode_location(coords).float()
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(
            dim=1, keepdim=True
        )

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_location


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16."""

    def _convert_weights_to_fp16(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()

        if isinstance(module, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(module, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(module, name):
                attr = getattr(module, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
