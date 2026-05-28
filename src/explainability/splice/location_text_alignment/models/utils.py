from models.layers import EmbeddingProjection
from models.location_encoder import LocationEncoder
from models.text_encoder import TEXT_EMBEDDING_DIMENSIONS, TextEncoder


def make_embedding_projection(
    in_dim: int,
    out_dim: int,
    projection_type: str,
    num_hidden_layers: int,
    num_hidden_features: int,
    nonlinearity: str,
):
    """
    Build EmbeddingProjection layer, or return None if projection_type is 'none'.

    projection_type: linear (single layer) or mlp (with hidden layers)
    """
    if projection_type == "none" or projection_type is None:
        return None

    if projection_type == "linear":
        return EmbeddingProjection(in_dim, out_dim, num_hidden_layers=0)

    if projection_type == "mlp":
        return EmbeddingProjection(
            in_dim,
            out_dim,
            num_hidden_layers=num_hidden_layers,
            num_hidden_features=num_hidden_features,
            nonlinearity=nonlinearity,
        )

    raise ValueError(f"Unknown projection type: {projection_type}")


def make_text_encoder(
    text_encoder_type: str,
    projection_type: str,
    out_dim: int,
    finetune_mode: str,
    num_hidden_layers: int,
    num_hidden_features: int,
    nonlinearity: str,
    precomputed: bool = True,
    **lora_kwargs,
) -> TextEncoder:
    """
    Build a TextEncoder with an optional projection into out_dim.
    """
    in_dim = TEXT_EMBEDDING_DIMENSIONS[text_encoder_type]
    proj = make_embedding_projection(
        in_dim,
        out_dim,
        projection_type,
        num_hidden_layers,
        num_hidden_features,
        nonlinearity,
    )
    return TextEncoder(
        text_encoder_type,
        embed_project=proj,
        finetune_mode=finetune_mode,
        precomputed=precomputed,
        **lora_kwargs,
    )


def make_location_encoder(
    location_encoder_type: str,
    finetune_mode: str,
    precomputed: bool = True,
) -> LocationEncoder:
    """
    Build a LocationEncoder with no projection (output dim = native embedding dim).
    """
    return LocationEncoder(
        location_encoder_type,
        embed_project=None,
        finetune_mode=finetune_mode,
        precomputed=precomputed,
    )
