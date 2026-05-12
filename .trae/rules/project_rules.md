# DSCLR 项目规则

## 运行环境规则

### CUDA 使用规则（重要！）
**所有评测和模型推理必须使用 CUDA/GPU 运行，禁止使用 CPU。**

原因：Trae IDE 的沙箱环境（`trae-sandbox`）会限制 CUDA 访问。必须直接在终端运行命令，绕过沙箱限制

正确做法：
```bash
# 直接在终端运行，不经过沙箱
cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.xxx --device=cuda
```

### 设备参数
- 默认使用 `--device=cuda` 或让代码自动检测
- 已修复 `engine_deir_dual_v2.py` 中的 CUDA 检测：先调用 `torch.cuda._lazy_init()` 再检查 `is_available()`

## 评测相关

### target_avg 定义（重要！）
**target_avg = (Core17_changed_MAP@1000 + Robust04_changed_MAP@1000 + News21_changed_nDCG@5) / 3**
除非用户特别说明，否则"平均指标"均指此定义，而非三个数据集 MAP 的简单平均。

### DeIR-Dual V2 最佳参数（截至 2026-04-25）

#### Repllama 编码器

**测试集网格搜索最优**：
- α=1.0, β=1.0, δ=0.0
- target_avg=0.26708
- 来源：results/repllama-v2-grid/

**训练集导出最优（学术可接受方法）**：
- α=0.5, β=0.8, δ=0.0（top-1000 retrieval-simulated 采样）
- 方法：训练集 + top-1000 retrieval-simulated 200干扰项 + V2公式网格搜索

#### Mistral (E5-Mistral-7B) 编码器

**测试集网格搜索最优**：
- α=0.1, β=1.1, δ=0.05
- 来源：results/mistral-v2-grid/

**训练集导出最优（学术可接受方法）**：
- α=0.3, β=1.0, δ=0.05（top-1000 retrieval-simulated 采样 compromise）
- 方法：训练集 + top-1000 retrieval-simulated 200干扰项 + V2公式网格搜索

#### 编码器无关参数搜索策略（EAPS）
1. **Retrieval-Simulated Distractor Sampling**：从所有负文档中按 S_base 降序取 top-k（k=1000），再从中采样 200 个干扰项
2. **关键洞察**：不同编码器的 at-risk 比例差异巨大
   - Mistral: 62.9% 负文档 S_neg > S_base，top-1000 at-risk=28.3%
   - Repllama: 0% 负文档 S_neg > S_base，top-1000 at-risk=0.08%
3. **top-k 选择**：k=1000 比 k=100 更好，因为更接近测试集的真实检索分布
4. **δ 方向**：高 at-risk 编码器（如 Mistral）需要正 δ 来限制惩罚范围
