# LLM-benchmark

统一集成多种 LLM / Agent 评测基准，便于在同一仓库内按需运行、对比与复现。

> 本仓库为 **benchmark 集成仓**：各子目录是相对独立的评测项目，依赖、配置与运行方式以各子目录 README 为准。

## 包含的基准

| 目录 | 基准 | 评测能力 | 文档 |
|------|------|----------|------|
| [`IFBench`](./IFBench) | IFBench | 精确指令遵循（OOD constraints） | [README](./IFBench/README.md) |
| [`hle`](./hle) | Humanity's Last Exam (HLE) | 前沿学科闭卷问答（多模态） | [README](./hle/README.md) |
| [`llm-eval`](./llm-eval) | LLM Eval | AIME26 / GPQA Diamond / NL2Repo 等 | [README](./llm-eval/README.md) |
| [`scicode`](./scicode) | SciCode | 科研代码生成与求解 | [README](./scicode/README.md) |
| [`swebench`](./swebench) | SWE-bench | 真实软件仓库缺陷修复 | [swebench-kit](./swebench/swebench-kit/README.md) |
| [`terminal-bench-2-1`](./terminal-bench-2-1) | Terminal-Bench 2.1 | 终端 / 容器内复杂任务 | [README](./terminal-bench-2-1/README.md) |
| [`gdpval-bench`](./gdpval-bench) | GDPval (AA-aligned) | 经济价值职业任务 agent 评测（生成 + judge） | [README](./gdpval-bench/README.md) |

## 仓库结构

```text
LLM-benchmark/
├── IFBench/                 # 指令遵循评测
├── hle/                     # Humanity's Last Exam
├── llm-eval/                # 通用 LLM 评测（含 sandbox 子项目）
├── scicode/                 # 科研代码基准
├── swebench/                # SWE-bench 评测脚本与工具链
│   └── swebench-kit/        # 共享评测台（配置驱动管线）
├── terminal-bench-2-1/      # Terminal-Bench 2.1 任务集
├── gdpval-bench/            # GDPval AA-aligned 评测
├── LICENSE                  # 根目录 Apache-2.0
└── README.md
```

## 快速开始

```bash
git clone https://github.com/Bruce-lys/LLM-benchmark.git
cd LLM-benchmark
```

选择要运行的基准，进入对应目录，按其 README 安装依赖并评测：

```bash
# 示例：LLM Eval
cd llm-eval
# 见 llm-eval/README.md

# 示例：IFBench
cd IFBench
# 见 IFBench/README.md

# 示例：SWE-bench（工具链入口）
cd swebench/swebench-kit
# 见 swebench-kit/README.md 与 RULES.md
```

> 各子项目依赖、API Key、沙箱要求不同，**请勿期望在根目录一次安装全部环境**。

## 环境说明

- 推荐按子项目分别使用虚拟环境（如 `uv` / `conda` / `venv`）。
- 需要独立沙箱的评测（coding / terminal / SWE-bench 等）按其子目录说明准备 Docker、Harbor 或其他运行时。
- API Key、模型端点、本地路径等敏感或机器相关配置不要提交进仓库；优先使用 `.env` / 本地 YAML（仓库中通常提供 `.env.example` 或 `config.example.yaml`）。

## 各基准入口速览

### IFBench

精确指令遵循评测。安装依赖后，用测试集与模型输出 jsonl 运行：

```bash
cd IFBench
pip install -r requirements.txt
python3 -m run_eval \
  --input_data=IFBench_test.jsonl \
  --input_response_data=sample_output.jsonl \
  --output_dir=eval
```

详情见 [`IFBench/README.md`](./IFBench/README.md)。

### HLE (Humanity's Last Exam)

前沿学科闭卷基准。数据集：[🤗 cais/hle](https://huggingface.co/datasets/cais/hle)。

```bash
cd hle
pip install -r requirements.txt
# 按 README 配置 OpenAI 兼容接口后运行评测
```

详情见 [`hle/README.md`](./hle/README.md)。

### LLM Eval

覆盖 AIME26、GPQA Diamond，以及需要沙箱的 NL2Repo 等：

```bash
cd llm-eval
uv sync
source .venv/bin/activate
```

详情见 [`llm-eval/README.md`](./llm-eval/README.md)。

### SciCode

科研场景代码生成评测：

```bash
cd scicode
# 按 README / pyproject 安装依赖后运行评测脚本
```

详情见 [`scicode/README.md`](./scicode/README.md)。

### SWE-bench

本目录提供评测脚本与 [`swebench-kit`](./swebench/swebench-kit/README.md)（YAML 配置驱动：起服务 → 生成轨迹 → 官方评分）。请先阅读 `swebench-kit/RULES.md`。

### Terminal-Bench 2.1

容器环境下的终端任务评测，推荐使用 [Harbor](https://github.com/laude-institute/harbor)：

```bash
uv tool install harbor
# 按 terminal-bench-2-1/README.md 提交或本地跑分
```

详情见 [`terminal-bench-2-1/README.md`](./terminal-bench-2-1/README.md)。

### GDPval-bench

OpenAI GDPval gold 子集的 AA-aligned 评测脚手架（Stirrup 生成 + 多模态 judge）：

```bash
cd gdpval-bench
uv sync
# 见 gdpval-bench/README.md
```

详情见 [`gdpval-bench/README.md`](./gdpval-bench/README.md)。

## 来源与致谢

本仓库整合了多个开源评测项目，原始工作归属于各自作者与机构。使用或发表结果时，请同时引用对应上游论文 / 仓库 / 数据集：

| 基准 | 上游参考 |
|------|----------|
| IFBench | [Paper](https://arxiv.org/pdf/2507.02833) · [HF Collection](https://huggingface.co/collections/allenai/ifbench-683f590687f61b512558cdf1) · [open-instruct IF-RLVR](https://github.com/allenai/open-instruct) |
| HLE | [Website](https://lastexam.ai) · [Paper](https://arxiv.org/abs/2501.14249) · [Dataset](https://huggingface.co/datasets/cais/hle) |
| SciCode | [Homepage](https://scicode-bench.github.io/) · [Paper](https://arxiv.org/abs/2407.13168) · [Dataset](https://huggingface.co/datasets/SciCode1/SciCode) |
| SWE-bench | [swebench.com](https://www.swebench.com/) · [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) |
| Terminal-Bench | [tbench.ai](https://www.tbench.ai) · [Leaderboard 2.1](https://www.tbench.ai/leaderboard/terminal-bench/2.1) |
| LLM Eval | 见 [`llm-eval/README.md`](./llm-eval/README.md) 中各数据集链接（AIME26 / GPQA / NL2Repo 等） |
| GDPval | [HF openai/gdpval](https://huggingface.co/datasets/openai/gdpval) · [AA methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking/) · [Stirrup](https://github.com/ArtificialAnalysis/Stirrup) |

## License

本仓库根目录采用 [Apache-2.0](./LICENSE)。

各子目录可能沿用其上游许可证，以子目录内 `LICENSE` 或上游声明为准。例如：

- `IFBench`、`scicode`、`terminal-bench-2-1`：Apache-2.0
- `hle`：MIT

## 贡献

欢迎通过 Issue / PR 补充新的基准目录，或改进统一文档与脚本。新增基准时建议：

1. 放入独立子目录，保留其原有 README 与许可证文件；
2. 在本 README 的「包含的基准」表格中登记；
3. 在「来源与致谢」中补充上游链接。
