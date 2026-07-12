"""GritLM encoder wrapper for retrieval experiments.

Uses AutoModel + AutoTokenizer directly to bypass the GritLM library's
bbcc attention implementation which produces NaN embeddings with
transformers >= 5.9.0 due to MistralConfig.rope_theta removal.

Key implementation notes:
- The Mistral tokenizer used by GritLM does NOT have <|embed|> as a
  real special token; it tokenizes as [<|, embed, |>]. Token ID 1 (BOS)
  is also the EOS/pad token, so padding_side='right' is required.
- GritLM passes is_causal=False when bbcc attention is used; we do the
  same for the standard Mistral attention.
- Query encoding uses instruction masking (embed_instruction=False):
  instruction tokens are included in the forward pass for context but
  masked out during mean pooling.
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from .encoder import BaseEncoder

logger = logging.getLogger(__name__)


class GritLMEncoder(BaseEncoder):
    """GritLM embedding interface using AutoModel directly.

    Bypasses the official `gritlm.GritLM` class which has a custom bbcc
    attention implementation incompatible with transformers >= 5.9.0.
    We replicate the same encoding pipeline: same tokenizer settings,
    instruction format, mean pooling with instruction masking, and
    normalization.
    """

    DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(
        self,
        model_name: str = "GritLM/GritLM-7B",
        device: str = "cuda",
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        **kwargs,
    ):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.modeling_utils import PreTrainedModel

        # Allow model hub access even if env vars say offline
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        try:
            import huggingface_hub.constants as hf_constants
            hf_constants.HF_HUB_OFFLINE = False
        except Exception:
            pass

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length or 512

        logger.info("Loading GritLM encoder (AutoModel): %s", model_name)

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Patch rope_theta for transformers >= 5.9.0 compat
        if not hasattr(config, "rope_theta"):
            config.rope_theta = 10000.0

        # Quantization config (optional)
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info("Using 4-bit quantization")

        # Suppress "UNEXPECTED" missing-key warnings
        original_initialize_missing_keys = PreTrainedModel._initialize_missing_keys
        PreTrainedModel._initialize_missing_keys = lambda self, is_quantized: None
        try:
            # Right-padding required: BOS token (id=1) doubles as pad/EOS,
            # left-padding would put BOS at the start of real content.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="right"
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = dict(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config

            if device_map and not quantization_config:
                # Multi-GPU: let accelerate split layers across visible GPUs
                model_kwargs["device_map"] = device_map
                self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
                # Use the device of the first parameter as self.device
                self.device = next(self.model.parameters()).device.type
                if ':' in str(next(self.model.parameters()).device):
                    self.device = str(next(self.model.parameters()).device)
            else:
                self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)
        finally:
            PreTrainedModel._initialize_missing_keys = original_initialize_missing_keys

        self.model.config.use_cache = False
        self.model.eval()
        logger.info("GritLM encoder loaded (AutoModel bypass)")

    @staticmethod
    def _instruction(instruction: str) -> str:
        """Format instruction in GritLM's expected prompt format.

        GritLM uses the character ⋃ (U+22C3) followed by the instruction
        and <|embed|> marker. The <|embed|> is NOT a real special token in
        the Mistral vocabulary; it tokenizes as [<|, embed, |>].

        Returns empty string for empty/None instruction (no instruction prefix).
        """
        if not instruction:
            return ""
        return f"⋃\n{instruction}\n<|embed|>\n"

    def _encode_batch(
        self,
        texts: List[str],
        instruction: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode texts with given instruction, returning normalized embeddings.

        Replicates GritLM's encoding pipeline:
        1. Prepend instruction to each text
        2. Mask out instruction tokens during mean pooling (embed_instruction=False)
        3. Normalize embeddings
        """
        formatted = [instruction + t for t in texts]
        all_embeddings = []

        for i in range(0, len(formatted), batch_size):
            batch = formatted[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state

            # Move attention mask to the same device as hidden (last layer may be on a different GPU)
            attn_mask = inputs["attention_mask"].clone().to(hidden.device)
            if instruction:
                instruction_tokens = self.tokenizer(
                    instruction,
                    padding=False,
                    truncation=True,
                    max_length=self.max_seq_length,
                )["input_ids"]
                attn_mask[:, : len(instruction_tokens)] = 0

            # Mean pooling over non-masked tokens
            mask = attn_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts

            if self.normalize_embeddings:
                mean_pooled = F.normalize(mean_pooled.float(), p=2, dim=1)
            else:
                mean_pooled = mean_pooled.float()

            all_embeddings.append(mean_pooled.cpu())

        return torch.cat(all_embeddings, dim=0).to(self.device)

    def encode_queries(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = batch_size or self.batch_size
        task = task or self.DEFAULT_TASK
        return self._encode_batch(texts, self._instruction(task), batch_size)

    def encode_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = batch_size or self.batch_size
        return self._encode_batch(texts, self._instruction(""), batch_size)

    def get_embedding_dim(self) -> int:
        return int(getattr(self.model.config, "hidden_size"))
