"""Vision-language model loading and caption generation (Molmo2-8B when supported, else BLIP)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Molmo2 expects recent transformers with AutoModelForImageTextToText; project pin may be older.
try:
    from transformers import AutoModelForImageTextToText as _ImageTextToTextModel
except ImportError:  # pragma: no cover
    _ImageTextToTextModel = None

try:
    from transformers import AutoProcessor, BlipForConditionalGeneration, BlipProcessor
except ImportError as e:  # pragma: no cover
    raise ImportError("transformers and PIL are required for vlm_model") from e


DEFAULT_MOLMO2_ID = "allenai/Molmo2-8B"


class VLMModel:
    """
    Loads **allenai/Molmo2-8B** when ``transformers`` exposes
    ``AutoModelForImageTextToText`` (>= ~4.57). Otherwise falls back to **BLIP**
    (``Salesforce/blip-image-captioning-base``) for development.

    Upgrade ``transformers`` for full Molmo2 support: ``pip install -U 'transformers>=4.57'``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MOLMO2_ID,
        *,
        device: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: bool = True,
        fallback_blip: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._backend: str = "molmo2"
        self.model: Any = None
        self.processor: Any = None

        if _ImageTextToTextModel is not None:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                )
                self.model = _ImageTextToTextModel.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                )
                self._backend = "molmo2"
                logger.info("Loaded Molmo2-style model %s", model_id)
                return
            except Exception as e:
                logger.warning("Molmo2 load failed (%s); trying fallback if enabled.", e)

        if not fallback_blip:
            raise RuntimeError(
                "Could not load Molmo2 and fallback_blip=False. "
                "Install a newer transformers (>=4.57) or set fallback_blip=True."
            )

        blip_id = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(blip_id)
        self.model = BlipForConditionalGeneration.from_pretrained(blip_id).to(self.device)
        self.model.eval()
        self._backend = "blip"
        self.model_id = blip_id
        logger.warning("Using BLIP fallback (%s); upgrade transformers for Molmo2-8B.", blip_id)

    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate a caption for a single RGB PIL image."""
        if self._backend == "molmo2":
            return self._generate_molmo2(image, prompt, max_new_tokens=max_new_tokens)
        return self._generate_blip(image, prompt, max_new_tokens=max_new_tokens)

    def _generate_molmo2(self, image: Image.Image, prompt: str, *, max_new_tokens: int) -> str:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        cut = inputs["input_ids"].size(1)
        generated_tokens = generated_ids[0, cut:]
        text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text.strip()

    def _generate_blip(self, image: Image.Image, prompt: str, *, max_new_tokens: int) -> str:
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_length=max_new_tokens)
        return self.processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
