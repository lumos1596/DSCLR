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

### pMRR: 衡量指令敏感度

### 技术细节和实验结果查询
需要查看 DeIR-Dual V2 的核心公式、最优参数、参数搜索策略等技术细节时，调用 `dsclr-tech-details` skill。如果实验结果有更新，需要及时更新到 `dsclr-exp-details` skill 中。

