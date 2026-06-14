# DeIR-Dual V2 论文写作项目

## 目标
投稿 **AAAI 2027**，题目：*DeIR-Dual: Dual-Query Instruction Rewriting for Training-Free Instruction-Following Retrieval*

## 目录结构
```
paper/
├── OUTLINE.md           # 论文骨架大纲（主要规划文件）
├── README.md            # 本文件
├── Makefile             # 编译脚本（make / make quick / make clean）
├── aaai26.sty           # AAAI 格式占位文件（⚠️ 需要替换为官方版本）
├── tex/
│   └── paper.tex        # LaTeX 主文件（骨架已搭建，待填充内容）
├── figures/             # 图表素材（待添加）
├── tables/              # 表格数据/代码（待添加）
└── refs/
    └── references.bib   # BibTeX 参考文献（含占位引用，待完善）
```

## 编译方法

```bash
# 完整编译（含参考文献）
cd /home/luwa/Documents/DSCLR/paper && make

# 快速编译（不更新参考文献）
cd /home/luwa/Documents/DSCLR/paper && make quick

# 清理中间文件
cd /home/luwa/Documents/DSCLR/paper && make clean
```

## ⚠️ 重要：替换 aaai26.sty

当前使用的是占位样式文件，实际编译前需要从 AAAI 官方下载最新版本：
1. 访问 https://github.com/aaai/aaai-template
2. 下载最新的 `aaai26.sty` 或 `aaai27.sty`
3. 替换 `paper/aaai26.sty`

## 进度

- [x] 论文大纲（OUTLINE.md）
- [ ] LaTeX 模板搭建
- [ ] §5 Experiments 撰写
- [ ] §3 Method 撰写
- [ ] §4 First-Principles 撰写
- [ ] §6 Analysis 撰写
- [ ] §2 Related Work 撰写
- [ ] §1 Introduction + Abstract 撰写
- [ ] 图表制作
- [ ] 文献检索与引用
- [ ] 模拟审稿

## 关键提醒
- **所有写作工作在此目录下完成**
- 数据来自 `/home/luwa/Documents/DSCLR/` 下的 results/ 和 SKILL.md
- 禁止编造结果、引用或实验数据
- AAAI 格式要求：正文 7 页 + References（需确认 2027 官方要求）
