"""
RepLLaMA / Promptriever 编码器
使用正确的 prompt template 和 last token 提取
支持两类 PEFT adapter：
1. castorini/repllama-v1-7b-lora-passage (原版 LoRA)
2. samaya-ai/promptriever-* (官方 Promptriever adapter)
"""

import os
import logging
from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from .encoder import BaseEncoder

logger = logging.getLogger(__name__)


class RepLLaMAEncoder(BaseEncoder):
    """RepLLaMA 编码器 - 使用正确的 prompt template"""
    
    REPRODUCED_MODEL = "samaya-ai/RepLLaMA-reproduced"
    ORIGINAL_LORA = "castorini/repllama-v1-7b-lora-passage"
    PROMPTRIEVER_PREFIX = "samaya-ai/promptriever"
    
    # 基础模型路径映射
    BASE_MODEL_MAP = {
        "llama2": "/home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf",
        "llama3": "/home/luwa/Documents/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "mistral": "/home/luwa/Documents/models/mistral/Mistral-7B-v0.1",
    }

    # 当本地路径不存在时，回退到 HF 镜像仓库 ID（非 gated，可公开下载）
    BASE_MODEL_HF_FALLBACK = {
        "llama3": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "llama2": "NousResearch/Llama-2-7b-hf",
    }
    
    # Adapter 本地路径映射
    ADAPTER_LOCAL_MAP = {
        "samaya-ai/promptriever-llama3.1-8b-instruct-v1": "/home/luwa/Documents/models/promptriever-llama3.1-8b-instruct-v1",
        "samaya-ai/promptriever-llama3.1-8b-v1": "/home/luwa/Documents/models/promptriever-llama3.1-8b-v1",
        "samaya-ai/promptriever-llama2-7b-v1": "/home/luwa/Documents/models/promptriever-llama2-7b-v1",
        "samaya-ai/RepLLaMA-reproduced": "/home/luwa/.cache/huggingface/hub/models--samaya-ai--RepLLaMA-reproduced/snapshots/7f7fd93984469df6790272ba6bea7b05bb3f319e",
    }
    
    def __init__(
        self,
        model_name: str = "samaya-ai/RepLLaMA-reproduced",
        device: str = "cuda",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        base_model_path: str = None
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length or 2600
        
        # 自动推断基础模型路径
        if base_model_path is not None:
            self.base_model_path = base_model_path
        else:
            self.base_model_path = self._infer_base_model(model_name)
        
        logger.info(f"📥 加载 RepLLaMA 模型...")
        logger.info(f"   基础模型: {self.base_model_path}")
        
        if model_name.startswith(self.PROMPTRIEVER_PREFIX):
            self._load_peft_adapter_model(model_name)
        elif model_name == self.REPRODUCED_MODEL:
            self._load_peft_adapter_model(self.REPRODUCED_MODEL)
        else:
            self._load_peft_adapter_model(model_name)
        
        logger.info(f"✅ RepLLaMA 模型加载完成")
    
    @classmethod
    def _infer_base_model(cls, model_name: str) -> str:
        """根据 adapter 名称推断基础模型（本地路径优先，缺失时回退到 HF 镜像仓库 ID）"""
        name_lower = model_name.lower()
        if "llama3" in name_lower or "llama-3" in name_lower or "llama3.1" in name_lower:
            local = cls.BASE_MODEL_MAP["llama3"]
            if os.path.isdir(local):
                return local
            fallback = cls.BASE_MODEL_HF_FALLBACK["llama3"]
            logger.info(f"   本地基础模型路径不存在: {local}，回退到 HF 镜像: {fallback}")
            return fallback
        elif "mistral" in name_lower:
            return cls.BASE_MODEL_MAP["mistral"]
        else:
            # 默认 LLaMA2
            local = cls.BASE_MODEL_MAP["llama2"]
            if os.path.isdir(local):
                return local
            fallback = cls.BASE_MODEL_HF_FALLBACK["llama2"]
            logger.info(f"   本地基础模型路径不存在: {local}，回退到 HF 镜像: {fallback}")
            return fallback
    
    def _load_peft_adapter_model(self, adapter_path: str):
        """加载 PEFT adapter（RepLLaMA 或 Promptriever）+ 本地基础模型"""
        # 解析 adapter 本地路径：优先本地，缺失则回退到 HF 仓库 ID（自动下载）
        local_adapter_path = self.ADAPTER_LOCAL_MAP.get(adapter_path, adapter_path)
        if not os.path.isdir(local_adapter_path):
            # 尝试在 models 目录下查找
            candidate = os.path.join("/home/luwa/Documents/models", adapter_path.split("/")[-1])
            if os.path.isdir(candidate):
                local_adapter_path = candidate
            else:
                # 本地不存在，回退到 HF 仓库 ID（允许 HF Hub 自动下载）
                local_adapter_path = adapter_path
                logger.info(f"   本地 adapter 不存在，使用 HF 仓库 ID: {adapter_path}")

        # 判断基础模型是本地路径还是 HF 仓库 ID
        is_base_model_local_path = os.path.isdir(self.base_model_path)

        logger.info(f"   加载 PEFT adapter: {adapter_path} -> {local_adapter_path}")
        logger.info(f"   基础模型: {self.base_model_path} ({'本地' if is_base_model_local_path else 'HF 仓库'})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        
        # 判断是否为 LLaMA3 基础模型（更大，需要多卡）
        is_llama3 = "llama-3" in self.base_model_path.lower() or "llama3" in self.base_model_path.lower()
        
        if is_llama3:
            # LLaMA3-8B: float16 + 多卡分布
            base_model = AutoModel.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            # LLaMA2-7B: 8-bit 量化
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            base_model = AutoModel.from_pretrained(
                self.base_model_path,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=torch.float16
            )
        
        model = PeftModel.from_pretrained(base_model, local_adapter_path)
        # 不 merge_and_unload：device_map="auto" 多卡分布下，合并需要将所有层
        # 临时加载到单 GPU，导致 OOM。直接使用 PEFT 模型推理即可。
        self.model = model
        self.model.eval()
        
        # 记录输入设备
        self.input_device = self.model.device
        logger.info(f"   模型输入设备: {self.input_device}")
        
        # 记录是否为 LLaMA3 版 Promptriever（编码方式不同）
        self.is_llama3_promptriever = is_llama3 and ("promptriever" in adapter_path.lower() or "promptriever" in local_adapter_path.lower())
    
    def _encode_with_template(self, texts: List[str], template: str, batch_size: Optional[int] = None, show_progress: bool = True) -> torch.Tensor:
        """使用 template 编码文本"""
        batch_size = batch_size or self.batch_size
        all_embeddings = []

        formatted_texts = [template.format(text=text) for text in texts]

        num_batches = (len(formatted_texts) + batch_size - 1) // batch_size
        progress_bar = tqdm(total=num_batches, desc="编码中", unit="batch", disable=not show_progress)

        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i + batch_size]

            if self.is_llama3_promptriever:
                # Promptriever-LLaMA3 官方方式: 追加 EOS token + EOS pooling
                batch_dict = self.tokenizer(
                    batch_texts,
                    max_length=self.max_seq_length - 1,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    padding=False,
                    truncation=True,
                )
                batch_dict["input_ids"] = [
                    ids + [self.tokenizer.eos_token_id]
                    for ids in batch_dict["input_ids"]
                ]
                inputs = self.tokenizer.pad(
                    batch_dict,
                    padding=True,
                    pad_to_multiple_of=8,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        last_hidden_state = outputs.last_hidden_state
                        sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
                        batch_size_cur = last_hidden_state.shape[0]
                        batch_embeddings = last_hidden_state[
                            torch.arange(batch_size_cur, device=last_hidden_state.device),
                            sequence_lengths,
                        ]
                        if self.normalize_embeddings:
                            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            else:
                # LLaMA2 版: last token pooling
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                ).to(self.input_device if hasattr(self, 'input_device') else self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                    batch_embeddings = []
                    for j in range(len(batch_texts)):
                        seq_len = inputs['attention_mask'][j].sum().item()
                        embedding = outputs.last_hidden_state[j, seq_len - 1]
                        batch_embeddings.append(embedding)

                    batch_embeddings = torch.stack(batch_embeddings)

                    if self.normalize_embeddings:
                        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            all_embeddings.append(batch_embeddings.cpu())
            progress_bar.update(1)

        progress_bar.close()

        return torch.cat(all_embeddings, dim=0)
    
    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码查询 - 使用 query: template"""
        if self.is_llama3_promptriever:
            # Promptriever-LLaMA3 官方: 双空格，不追加 </s>（EOS token 在 _encode_with_template 中追加）
            return self._encode_with_template(texts, "query:  {text}", batch_size)
        return self._encode_with_template(texts, "query: {text}</s>", batch_size)
    
    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码文档 - 使用 passage: template"""
        if self.is_llama3_promptriever:
            # Promptriever-LLaMA3 官方: 双空格，不追加 </s>
            return self._encode_with_template(texts, "passage:  {text}", batch_size)
        return self._encode_with_template(texts, "passage: {text}</s>", batch_size)
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return 4096
