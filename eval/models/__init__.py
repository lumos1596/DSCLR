"""模型模块"""
from .encoder import BaseEncoder, SentenceTransformerEncoder, ModelFactory, DenseRetriever
from .e5_mistral_encoder import E5MistralEncoder
from .repllama_encoder import RepLLaMAEncoder
from .llama3_encoder import LLaMA31Encoder
from .infx_encoder import INFXRetrieverEncoder
from .gritlm_encoder import GritLMEncoder
from .bge_encoder import BGEEncoder

# 注册 E5-Mistral 编码器
ModelFactory.register("e5-mistral-7b-instruct", E5MistralEncoder)
ModelFactory.register("intfloat/e5-mistral-7b-instruct", E5MistralEncoder)

# 注册 RepLLaMA 编码器
ModelFactory.register("repllama-v1-7b-lora-passage", RepLLaMAEncoder)
ModelFactory.register("castorini/repllama-v1-7b-lora-passage", RepLLaMAEncoder)
ModelFactory.register("samaya-ai/promptriever-llama2-7b-v1", RepLLaMAEncoder)
ModelFactory.register("samaya-ai/promptriever-mistral-v0.1-7b-v1", RepLLaMAEncoder)
ModelFactory.register("samaya-ai/promptriever-llama3.1-8b-v1", RepLLaMAEncoder)
ModelFactory.register("samaya-ai/promptriever-llama3.1-8b-instruct-v1", RepLLaMAEncoder)

# 注册 LLaMA 3.1 8B Instruct 编码器
ModelFactory.register("Meta-Llama-3.1-8B-Instruct", LLaMA31Encoder)
ModelFactory.register("meta-llama/Meta-Llama-3.1-8B-Instruct", LLaMA31Encoder)

# 注册 INF-X-Retriever 编码器
ModelFactory.register("inf-retriever-v1-pro", INFXRetrieverEncoder)
ModelFactory.register("infly/inf-retriever-v1-pro", INFXRetrieverEncoder)

# 注册 GritLM 编码器
ModelFactory.register("GritLM/GritLM-7B", GritLMEncoder)
ModelFactory.register("gritlm-7b", GritLMEncoder)

# 注册 BGE 编码器
ModelFactory.register("BAAI/bge-large-en-v1.5", BGEEncoder)
ModelFactory.register("bge-large-en-v1.5", BGEEncoder)
ModelFactory.register("BGE-large-en-v1.5", BGEEncoder)

# 注册 BGE-M3 编码器 (多语言，无查询前缀)
from .bge_m3_encoder import BGEM3Encoder
ModelFactory.register("BAAI/bge-m3", BGEM3Encoder)
ModelFactory.register("bge-m3", BGEM3Encoder)

# 注册 BGE-base 编码器 (与 BGE-large 相同接口，查询前缀)
ModelFactory.register("BAAI/bge-base-en-v1.5", BGEEncoder)
ModelFactory.register("bge-base-en-v1.5", BGEEncoder)

# 注册 BGE-small 编码器 (与 BGE-large 相同接口，查询前缀)
ModelFactory.register("BAAI/bge-small-en-v1.5", BGEEncoder)
ModelFactory.register("bge-small-en-v1.5", BGEEncoder)

__all__ = [
    'BaseEncoder',
    'SentenceTransformerEncoder',
    'E5MistralEncoder',
    'RepLLaMAEncoder',
    'LLaMA31Encoder',
    'INFXRetrieverEncoder',
    'GritLMEncoder',
    'BGEEncoder',
    'ModelFactory',
    'DenseRetriever'
]
