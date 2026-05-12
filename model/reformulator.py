"""
DSCLR Query Reformulator Module
调用 LLM API 对查询进行解耦，返回 Q_plus 和 Q_minus
支持缓存和复用机制
"""

import os
import json
import logging
from typing import Tuple, Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_VERSION = "v4"

# 这里解锁APIkey
# API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-961483e59c904f65b619248417e0a06c")

SYSTEM_PROMPT_V4 = """You are an expert Information Retrieval (IR) query optimizer for Dense Vector Models.
Your task is to decouple a user's "Original Query" and complex "Instruction" into two concise, NATURAL LANGUAGE representations: Q_plus and Q_minus.

⚠️ CRITICAL RULES FOR Q_plus (Positive Intent):
1. Write a fluent, natural language phrase (e.g., "The effect of mobile phones on brain cancer incidence"). DO NOT output a robotic list of keywords.
2. Semantic Topic ONLY. Strictly REMOVE all format-related noise (e.g., discard words like "articles", "documents", "news", "statistical studies", "experiments", "mentions", "reports").
3. Remove all conversational connectors (e.g., "such as", "including new measures like", "specifically from"). Keep it a pure, fluent semantic query.
4. Keep it sharp, grammatically coherent, and focused on the core topic.

⚠️ CRITICAL RULES FOR Q_minus (Negative Constraints):
1. Extract ONLY the most critical, high-level concepts to exclude. Do not create a massive, disjointed list of words.
2. Use concise natural phrasing (e.g., "leukemia", "normal security screening methods", "land-bound sciences").
3. If no explicit exclusion exists, output exactly "[NONE]".

⚠️ CRITICAL RULES FOR Q_minus (The "No-Negation" Rule):
1. DENSE RETRIEVERS DO NOT UNDERSTAND LOGIC WORDS. You MUST NOT use negation words in Q_minus (e.g., "no", "not", "non-", "without", "outside", "other than").
2. Convert logical negations into AFFIRMATIVE ENTITIES.
   - BAD: "non-United Kingdom" -> GOOD: "United States, France, global"
   - BAD: "violence outside Ireland" -> GOOD: "violence in England, international violence"
   - BAD: "production without export" -> GOOD: "domestic production only"
3. Output ONLY the exact entities/concepts to be excluded.

Output strictly in JSON format:
{
  "Reasoning_Steps": "1. Identify core topic. 2. Filter format noise. 3. Identify core exclusions (convert negations to affirmatives).",
  "Q_plus": "Fluent, concise natural language query",
  "Q_minus": "Concise core exclusions or [NONE]"
}

---
EXAMPLES:

[Example 1: Removing Format Noise & Exact Negative]
Original Query: Evidence that radio waves from radio towers or car (mobile) phones affect brain cancer occurrence.
Instruction: Persons living near radio towers... A relevant document includes any experiment with animals, statistical study, articles, news items which report on the incidence... Any mentions of leukemia is not relevant.
Output:
{
  "Reasoning_Steps": "1. Topic: radio waves from towers/phones and brain cancer. 2. Noise: 'experiment with animals, statistical study, articles, news items'. 3. Negative: leukemia.",
  "Q_plus": "Association between radio waves from towers or mobile phones and brain cancer incidence",
  "Q_minus": "leukemia"
}

[Example 2: Condensing the "Kitchen Sink" Negative]
Original Query: Identify documents discussing the development and application of space-borne ocean remote sensing.
Instruction: Documents discussing... in oceanography, seabed prospecting are relevant. Documents that discuss the application of satellite remote sensing in geography, agriculture, forestry, mining and mineral prospecting or any land-bound science are not relevant, nor are references to international marketing or promotional advertising... Information about temperature is not relevant.
Output:
{
  "Reasoning_Steps": "1. Topic: space-borne ocean remote sensing in marine sciences. 2. Noise: 'Identify documents discussing', connectors like 'such as'. 3. Negative: Condense land-bound sciences and marketing (no negation words).",
  "Q_plus": "Development and application of space-borne ocean remote sensing in oceanography and seabed prospecting",
  "Q_minus": "land-bound sciences, agriculture, marketing, temperature"
}

[Example 3: Handling Negation Conversion & Fluent Q+]
Original Query: Find documents which describe an advantage in hiring potential or increased income for graduates of U.S. colleges.
Instruction: Relevant documents cite some advantage... Documents citing better opportunities for non-college vocational-training is not relevant. I do not trust colleges in Manhattan as they have a different economy, so find documents that do not cite or discuss these colleges (e.g. Columbia, etc.).
Output:
{
  "Reasoning_Steps": "1. Topic: hiring potential and income advantages for US college graduates. 2. Noise: 'Find documents which describe', 'I do not trust'. 3. Negative: Convert 'non-college vocational training' to 'college vocational training only' and 'Manhattan colleges like Columbia' stays.",
  "Q_plus": "Advantage in hiring potential and increased income for United States college graduates",
  "Q_minus": "college vocational training only, Manhattan colleges such as Columbia"
}

[Example 4: The NO Negative Case]
Original Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion of the project, its estimated or actual total cost...
Output:
{
  "Reasoning_Steps": "1. Topic: ongoing status of Three Gorges Project including completion date and cost. 2. Noise: 'A relevant document will provide'. 3. Negative: No exclusion mentioned.",
  "Q_plus": "Ongoing status of the Three Gorges Project including projected completion date, total cost, and electrical output",
  "Q_minus": "[NONE]"
}

Output in English."""

USER_PROMPT_TEMPLATE_V4 = """
请深吸一口气，运用你严密的逻辑分析能力，对以下 Query 和 Instruction 进行抽取式提纯解耦：

输入:
Query: "{query}"
Instruction: "{instruction}"
"""

PROMPT_LIBRARY = {
    "v4": {
        "system": SYSTEM_PROMPT_V4,
        "user_template": USER_PROMPT_TEMPLATE_V4,
    }
}


def get_prompt_templates(prompt_version: str) -> Tuple[str, str]:
    """根据版本号返回 system prompt 和 user prompt template。"""
    version = (prompt_version or DEFAULT_PROMPT_VERSION).lower()
    if version not in PROMPT_LIBRARY:
        available = ", ".join(sorted(PROMPT_LIBRARY.keys()))
        raise ValueError(f"不支持的 prompt_version: {prompt_version}. 可用版本: {available}")
    prompt_conf = PROMPT_LIBRARY[version]
    return prompt_conf["system"], prompt_conf["user_template"]


DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_BACKOFF_FACTOR = 2.0


class APIRetryError(Exception):
    """API 重试次数耗尽异常"""
    def __init__(self, message: str, attempts: int, last_error: str):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def call_llm_api(
    query: str,
    instruction: str,
    api_key: str = API_KEY,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    on_retry: Optional[callable] = None
) -> Tuple[str, str]:
    """
    调用 LLM API 进行查询解耦
    
    Args:
        query: 查询文本
        instruction: 指令文本
        api_key: API 密钥
        prompt_version: 提示词版本，默认使用 v4
        max_retries: 最大重试次数
        initial_delay: 初始重试延迟（秒）
        max_delay: 最大重试延迟（秒）
        backoff_factor: 退避因子
        on_retry: 重试回调函数 (attempt: int, error: str) -> None
        
    Returns:
        (Q_plus, Q_minus) 元组
        
    Raises:
        APIRetryError: 当重试次数耗尽时抛出
    """
    from utils.call_llm.call_deepseek import call_deepseek
    import time

    system_prompt, user_prompt_template = get_prompt_templates(prompt_version)
    
    user_prompt = user_prompt_template.format(
        query=query,
        instruction=instruction
    )
    
    last_error = "未知错误"
    delay = initial_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            result_text = call_deepseek(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                is_json=True,
                temperature=0.1
            )
            
            if not result_text:
                last_error = "API 返回为空"
                logger.warning(f"尝试 {attempt}/{max_retries}: API 返回为空")
                
                if attempt < max_retries:
                    if on_retry:
                        on_retry(attempt, last_error)
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                    continue
                else:
                    raise APIRetryError(
                        f"API 重试 {max_retries} 次后仍失败",
                        attempts=max_retries,
                        last_error=last_error
                    )
            
            parsed_json = json.loads(result_text)
            q_plus = parsed_json.get("Q_plus", "")
            q_minus = parsed_json.get("Q_minus", "[NONE]")

            if not q_plus:
                q_plus = query
            
            if attempt > 1:
                logger.info(f"API 调用成功 (重试 {attempt - 1} 次后)")
            
            return (q_plus, q_minus)
            
        except json.JSONDecodeError as e:
            last_error = f"JSON 解析失败: {e}"
            logger.warning(f"尝试 {attempt}/{max_retries}: {last_error}")
            
            if attempt < max_retries:
                if on_retry:
                    on_retry(attempt, last_error)
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                raise APIRetryError(
                    f"API 重试 {max_retries} 次后仍失败",
                    attempts=max_retries,
                    last_error=last_error
                )
                
        except Exception as e:
            last_error = f"API 调用失败: {e}"
            logger.warning(f"尝试 {attempt}/{max_retries}: {last_error}")
            
            if attempt < max_retries:
                if on_retry:
                    on_retry(attempt, last_error)
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                raise APIRetryError(
                    f"API 重试 {max_retries} 次后仍失败",
                    attempts=max_retries,
                    last_error=last_error
                )
    
    raise APIRetryError(
        f"API 重试 {max_retries} 次后仍失败",
        attempts=max_retries,
        last_error=last_error
    )


class DualQueryCache:
    """
    双流查询缓存管理器
    支持多任务区分、版本控制、高效查询
    """
    
    def __init__(self, cache_dir: str = "dataset/FollowIR_test/dual_queries_v4"):
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_file(self, task_name: str) -> str:
        """获取指定任务的缓存文件路径"""
        import glob
        pattern = os.path.join(self.cache_dir, f"dual_queries_*_{task_name}.jsonl")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        return os.path.join(self.cache_dir, f"dual_queries_v4_{task_name}.jsonl")
    
    def _get_version_file(self, task_name: str) -> str:
        """获取版本信息文件路径"""
        return os.path.join(self.cache_dir, f"version_{task_name}.json")
    
    def load_cache(self, task_name: str) -> Dict[str, Dict]:
        """加载指定任务的缓存数据"""
        cache_file = self._get_cache_file(task_name)
        cache_data = {}
        
        if not os.path.exists(cache_file):
            return cache_data
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    qid = item.get('qid')
                    query_type = item.get('query_type', 'og')
                    prompt_version = item.get('prompt_version', 'legacy')
                    if qid:
                        # 使用与查找时一致的键格式: f"{qid}_{query_type}_{prompt_version}"
                        cache_key = f"{qid}_{query_type}_{prompt_version}"
                        cache_data[cache_key] = item
        
        logger.info(f"✅ 已加载 {len(cache_data)} 条缓存 (task: {task_name})")
        return cache_data
    
    def save_record(self, task_name: str, record: Dict):
        """保存单条记录"""
        cache_file = self._get_cache_file(task_name)
        
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
    
    def save_batch(self, task_name: str, records: List[Dict]):
        """批量保存记录"""
        if not records:
            return
        
        cache_file = self._get_cache_file(task_name)
        
        with open(cache_file, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
        
        logger.info(f"💾 已保存 {len(records)} 条记录 (task: {task_name})")
    
    def get_record(self, task_name: str, qid: str) -> Optional[Dict]:
        """获取单条缓存记录"""
        cache = self.load_cache(task_name)
        return cache.get(qid)
    
    def get_all_records(self, task_name: str) -> Dict[str, Dict]:
        """获取所有缓存记录"""
        return self.load_cache(task_name)
    
    def clear_cache(self, task_name: str):
        """清空指定任务的缓存"""
        cache_file = self._get_cache_file(task_name)
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info(f"🗑️ 已清空缓存 (task: {task_name})")
    
    def get_version_info(self, task_name: str) -> Dict:
        """获取版本信息"""
        version_file = self._get_version_file(task_name)
        if os.path.exists(version_file):
            with open(version_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"version": "1.0", "created_at": None, "updated_at": None}
    
    def update_version(self, task_name: str, info: Dict):
        """更新版本信息"""
        version_file = self._get_version_file(task_name)
        info['updated_at'] = datetime.now().isoformat()
        
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)


class QueryReformulator:
    """
    查询重构器 - 封装 LLM API 调用
    支持缓存、失败记录和可配置的重试机制
    """
    
    def __init__(
        self,
        task_name: str = "Core17InstructionRetrieval",
        api_key: str = API_KEY,
        use_cache: bool = True,
        cache_dir: str = "dataset/FollowIR_test/dual_queries_v4",
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        log_failed_dir: str = "dataset/FollowIR_test/failed_queries"
    ):
        """
        初始化查询重构器
        
        Args:
            task_name: 任务名称 (用于区分不同数据集)
            api_key: DeepSeek API 密钥
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            prompt_version: 提示词版本，默认 v4
            max_retries: 最大重试次数
            initial_delay: 初始重试延迟（秒）
            max_delay: 最大重试延迟（秒）
            backoff_factor: 退避因子
            log_failed_dir: 失败记录目录
        """
        self.task_name = task_name
        self.api_key = api_key
        self.use_cache = use_cache
        self.prompt_version = (prompt_version or DEFAULT_PROMPT_VERSION).lower()
        self.cache = DualQueryCache(cache_dir)
        self._cache_data: Dict[str, Dict] = {}
        
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        self.log_failed_dir = log_failed_dir
        self._failed_records: List[Dict] = []
        self._retry_callback_attempts: Dict[str, int] = {}
        
        self._init_failed_logger()
        
        if self.use_cache:
            self._cache_data = self.cache.get_all_records(task_name)
    
    def _init_failed_logger(self):
        """初始化失败记录日志"""
        os.makedirs(self.log_failed_dir, exist_ok=True)
        self._failed_log_file = os.path.join(
            self.log_failed_dir,
            f"failed_{self.task_name}.jsonl"
        )
    
    def _on_retry(self, qid_key: str, attempt: int, error: str):
        """重试回调函数"""
        self._retry_callback_attempts[qid_key] = attempt
    
    def _log_failed_query(
        self,
        qid: str,
        idx: int,
        query: str,
        instruction: str,
        query_type: str,
        attempts: int,
        error: str
    ):
        """记录失败查询详情"""
        failed_record = {
            "task_name": self.task_name,
            "qid": qid,
            "idx": idx,
            "query": query,
            "instruction": instruction,
            "query_type": query_type,
            "attempts": attempts,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self._failed_records.append(failed_record)
        
        with open(self._failed_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(failed_record, ensure_ascii=False) + '\n')
        
        logger.error(
            f"❌ 记录失败查询: {qid} ({query_type}), "
            f"尝试 {attempts} 次, 错误: {error}"
        )
    
    def get_failed_summary(self) -> Dict:
        """获取失败统计摘要"""
        return {
            "total_failed": len(self._failed_records),
            "task_name": self.task_name,
            "log_file": self._failed_log_file
        }
    
    def _create_record(
        self,
        qid: str,
        idx: int,
        query: str,
        instruction: str,
        query_type: str,
        q_plus: str,
        q_minus: str
    ) -> Dict:
        """创建缓存记录"""
        return {
            "task_name": self.task_name,
            "qid": qid,
            "idx": idx,
            "query": query,
            "instruction": instruction,
            "query_type": query_type,
            "prompt_version": self.prompt_version,
            "q_plus": q_plus,
            "q_minus": q_minus,
            "created_at": datetime.now().isoformat()
        }
    
    def reformulate(
        self,
        qid: str,
        idx: int,
        query: str,
        instruction: str,
        query_type: str = "og"
    ) -> Tuple[str, str]:
        """
        对查询进行解耦重构
        
        Args:
            qid: 查询ID
            idx: 查询索引 (用于关联原数据集)
            query: 查询文本
            instruction: 指令文本
            query_type: 查询类型 (og/changed)
            
        Returns:
            (Q_plus, Q_minus) 元组
        """
        cache_key = f"{qid}_{query_type}_{self.prompt_version}"
        
        if self.use_cache and cache_key in self._cache_data:
            record = self._cache_data[cache_key]
            logger.debug(f"📥 使用缓存: {qid} ({query_type})")
            return (record['q_plus'], record['q_minus'])
        
        retry_callback = lambda attempt, error: self._on_retry(cache_key, attempt, error)
        
        try:
            q_plus, q_minus = call_llm_api(
                query, instruction, self.api_key,
                prompt_version=self.prompt_version,
                max_retries=self.max_retries,
                initial_delay=self.initial_delay,
                max_delay=self.max_delay,
                backoff_factor=self.backoff_factor,
                on_retry=retry_callback
            )
        except APIRetryError as e:
            logger.warning(f"API 重试耗尽，记录失败查询: {qid} ({query_type})")
            self._log_failed_query(
                qid=qid,
                idx=idx,
                query=query,
                instruction=instruction,
                query_type=query_type,
                attempts=e.attempts,
                error=e.last_error
            )
            return (query, "[NONE]")
        
        if self.use_cache:
            record = self._create_record(qid, idx, query, instruction, query_type, q_plus, q_minus)
            self._cache_data[cache_key] = record
            self.cache.save_record(self.task_name, record)
        
        return (q_plus, q_minus)
    
    def reformulate_batch(
        self,
        queries: List[Tuple[str, int, str, str, str]]
    ) -> Dict[str, Tuple[str, str]]:
        """
        批量解耦查询
        
        Args:
            queries: [(qid, idx, query, instruction, query_type), ...]
            
        Returns:
            {f"{qid}_{query_type}_{prompt_version}": (Q_plus, Q_minus), ...}
        """
        results = {}
        to_api_call = []
        
        for qid, idx, query, instruction, query_type in queries:
            cache_key = f"{qid}_{query_type}_{self.prompt_version}"
            
            if self.use_cache and cache_key in self._cache_data:
                record = self._cache_data[cache_key]
                results[cache_key] = (record['q_plus'], record['q_minus'])
            else:
                to_api_call.append((qid, idx, query, instruction, query_type))
        
        if to_api_call:
            logger.info(f"🔄 需要调用 API: {len(to_api_call)} 条")
            
            new_records = []
            for qid, idx, query, instruction, query_type in to_api_call:
                retry_callback = lambda attempt, error: self._on_retry(
                    f"{qid}_{query_type}_{self.prompt_version}", attempt, error
                )
                
                try:
                    q_plus, q_minus = call_llm_api(
                        query, instruction, self.api_key,
                        prompt_version=self.prompt_version,
                        max_retries=self.max_retries,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        backoff_factor=self.backoff_factor,
                        on_retry=retry_callback
                    )
                except APIRetryError as e:
                    logger.warning(f"API 重试耗尽，记录失败查询: {qid} ({query_type})")
                    self._log_failed_query(
                        qid=qid,
                        idx=idx,
                        query=query,
                        instruction=instruction,
                        query_type=query_type,
                        attempts=e.attempts,
                        error=e.last_error
                    )
                    q_plus, q_minus = query, "[NONE]"
                
                cache_key = f"{qid}_{query_type}_{self.prompt_version}"
                results[cache_key] = (q_plus, q_minus)
                
                if self.use_cache:
                    record = self._create_record(qid, idx, query, instruction, query_type, q_plus, q_minus)
                    new_records.append(record)
            
            if self.use_cache and new_records:
                self.cache.save_batch(self.task_name, new_records)
                for record in new_records:
                    pver = record.get('prompt_version', self.prompt_version)
                    cache_key = f"{record['qid']}_{record['query_type']}_{pver}"
                    self._cache_data[cache_key] = record
        
        return results
    
    def get_cached(self, qid: str, query_type: str = "og") -> Optional[Tuple[str, str]]:
        """获取缓存的查询结果"""
        cache_key = f"{qid}_{query_type}_{self.prompt_version}"
        if cache_key in self._cache_data:
            record = self._cache_data[cache_key]
            return (record['q_plus'], record['q_minus'])
        return None
    
    def has_cache(self, qid: str, query_type: str = "og") -> bool:
        """检查是否有缓存"""
        cache_key = f"{qid}_{query_type}_{self.prompt_version}"
        return cache_key in self._cache_data
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            "total": len(self._cache_data),
            "task": self.task_name
        }
    
    def clear_cache(self):
        """清空当前任务的缓存"""
        self.cache.clear_cache(self.task_name)
        self._cache_data = {}


def get_reformulator(
    task_name: str = "Core17InstructionRetrieval",
    api_key: str = API_KEY,
    use_cache: bool = True,
    prompt_version: str = DEFAULT_PROMPT_VERSION
) -> QueryReformulator:
    """
    获取查询重构器的便捷函数
    """
    return QueryReformulator(
        task_name=task_name,
        api_key=api_key,
        use_cache=use_cache,
        prompt_version=prompt_version
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    reformulator = get_reformulator(task_name="Core17InstructionRetrieval", use_cache=True)
    
    test_qid = "1-og"
    test_idx = 1
    test_query = "What are the causes of railway accidents throughout history?"
    test_instruction = "A relevant document should discuss the various factors contributing to railway accidents."
    
    print("测试 reformulate:")
    q_plus, q_minus = reformulator.reformulate(test_qid, test_idx, test_query, test_instruction, "og")
    print(f"  Q_plus: {q_plus}")
    print(f"  Q_minus: {q_minus}")
    
    print(f"\n缓存统计: {reformulator.get_cache_stats()}")
