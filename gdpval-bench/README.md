# GDPval-bench（AA-aligned）

基于 OpenAI [GDPval](https://huggingface.co/datasets/openai/gdpval) 公开 gold 子集的 agent 评测脚手架，生成侧使用 [Stirrup](https://github.com/ArtificialAnalysis/Stirrup)，judge 侧支持 cand-vs-human 的多模态两两对比。

> 本实现为 **AA-aligned 近似**：默认单评委 + vs-human win-rate，与 Artificial Analysis 官方 GDPval-AA v2（三评委面板 + model-vs-model Elo）不完全等价。差异见下方「与官方差异」。

## 快速开始

```bash
cd gdpval-bench
uv sync
# 可选：任务 code_exec 常用科学计算库
uv sync --extra task-runtime
```

系统依赖（judge 渲染 office 文档需要）：

- LibreOffice（`soffice`）
- poppler（`pdftoppm`）

## 题集

| 文件 | 题数 | 说明 |
|------|------|------|
| `config.json` | 220 | 原版全量 |
| `config_no_brave.json` | 206 | 去掉 14 道强依赖 web_search 的题 |

任务数据在 `tasks/<id>/`：

- `prompt.txt`
- `refs/`（参考附件，可空）
- `gold/`（人类专家交付物）

大媒体文件通过 Git LFS 跟踪（`*.mp4` / `*.wav` / 部分 zip）。克隆后请确保已安装并拉取 LFS：

```bash
git lfs install
git lfs pull
```

## 配置与运行

一模型一份 YAML（可从模板复制）：

```bash
cp aligned_TEMPLATE.yaml aligned_MyModel.yaml
# 编辑 candidate.model / base_url / sampling / tasks_file / brave_api_key / judge.api_key
```

生成：

```bash
uv run python -u gen_aligned.py --model MyModel
# 或
uv run python -u gen_aligned.py --config aligned_MyModel.yaml
```

Judge（materials-rich：完整 prompt + refs + zip 展开；仍为 gpt-5.5 + cand-vs-human）：

```bash
uv run python -u judge_aligned_gpt55.py --model MyModel
```

产物默认写到：

- `tasks/<id>/cand_<model>_aligned/s{0,1,2}/`
- `judge_aligned_<model>_*.json`（由 yaml `judge.results_file` 决定）

## 与官方差异（摘要）

| 项 | 本仓库默认 | AA GDPval-AA v2 |
|----|------------|-----------------|
| 执行环境 | 本地 LocalCodeExec | E2B 沙箱 |
| 搜索 | 可选 Brave；无 key 时用 web_fetch | Brave web_search |
| max turns | 配置项（常用 22） | 250 |
| 重复 | 常用 3 samples | Index 口径常 1 run |
| 评委 | 单模型（如 gpt-5.5） | 三面板随机抽样 |
| 比较 / 指标 | cand vs human win-rate | model vs model → Elo（human=1000） |

## 上游致谢

- OpenAI GDPval dataset: https://huggingface.co/datasets/openai/gdpval
- Artificial Analysis Stirrup / GDPval-AA methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking/
