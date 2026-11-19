# 模型推理项目结构规范

本文档是所有贡献者都必须遵守的项目结构和运行规范。目标是：每个人只需要把自己负责的模型放入正确的任务和个人目录，并在模型目录内维护完整流程，不需要适配中央调度流程。

## 项目整体结构

```text
model inference for GPU adaptation/
├── README.md
├── dataset/
│   └── <共享数据集和标注>
├── classification/
│   ├── evaluate.py
│   └── <contributor>/
│       ├── environment/
│       │   ├── README.md
│       │   ├── <env-name>.yml 或 <env-name>.txt
│       │   └── model_environment.csv
│       └── models/
│           └── <model-name>/
│               ├── inference.py
│               ├── inference/
│               │   └── <模型原生推理流程和模型专用辅助脚本>
│               ├── weights/
│               │   └── <模型权重或权重说明>
│               ├── result/
│               │   └── <推理结果和运行记录>
│               └── README.md
└── detection/
    ├── evaluate.py
    └── <contributor>/
        ├── environment/
        │   ├── README.md
        │   ├── <env-name>.yml 或 <env-name>.txt
        │   └── model_environment.csv
        └── models/
            └── <model-name>/
                ├── inference.py
                ├── inference/
                │   └── <模型原生推理流程和模型专用辅助脚本>
                ├── weights/
                │   └── <模型权重或权重说明>
                ├── result/
                │   └── <推理结果和运行记录>
                └── README.md
```

`classification/<contributor>/` 和 `detection/<contributor>/` 用于区分不同贡献者负责的模型。贡献者名称必须保留在目录路径中，不能把所有人的模型重新放进一个全局 `models/` 目录。

项目根目录的 `dataset/` 是共享数据集目录。共享数据集不需要复制到每个模型目录。如果某个模型需要额外的小型清单或预处理元数据，可以放在该模型目录内。

每个贡献者目录下必须有 `environment/`，用于记录该贡献者模型的运行依赖。依赖文件只记录环境，不把环境目录当作模型代码或权重目录。

## 环境依赖规范

每个贡献者必须在自己的 `environment/` 下维护：

- `README.md`：写明创建环境、激活环境和运行模型的命令。
- 至少一个依赖清单：使用 Conda 时为 `<env-name>.yml`，使用 pip 时为 `<env-name>.txt`。
- `model_environment.csv`：列出模型目录、使用的环境文件、Python 版本、GPU/运行时要求和已知限制。

如果该贡献者的所有模型可以使用同一套环境，只保留一个依赖清单，并在 `model_environment.csv` 中让所有模型指向它。如果需要两套或更多环境，必须为每套环境分别提供依赖清单，并在 `model_environment.csv` 中逐模型写清楚使用哪一套；不能只写“见 README”或依赖个人机器上的未记录环境。

环境依赖文件可以先记录已经验证过的包和版本；后续统一环境时再合并。新增模型在没有更新对应依赖清单和模型映射前，不算完成项目接入。

## 模型目录的硬性要求

每个模型目录必须至少包含：

```text
<model-name>/
├── inference.py
├── weights/
└── result/
```

### `inference.py`

`inference.py` 是所有模型统一的推理入口。无论模型原生流程中实际包含多少个脚本，最终都必须由这个文件启动完整推理流程。

### `weights/`

存放该模型实际使用的权重。若权重由外部挂载路径提供，必须在模型 README 中明确记录权重位置和使用方式。

### `result/`

存放该模型的推理结果、日志、状态文件和运行记录。模型不能把结果写入其他模型的目录。

### `inference/`

`inference/` 是模型专属推理代码目录，建议保留。可以放置：

- 模型原生推理脚本
- 数据预处理和后处理脚本
- 输出格式转换代码
- 计时代码
- 模型专用运行时辅助代码

这些代码必须属于模型自身，不要为了统一外观而抽到中央调度器中。

## 推理入口规范

每个模型的 `inference.py` 必须满足以下要求：

1. 使用相对项目路径或基于 `__file__` 的路径解析，不能依赖某个人电脑上的绝对路径。
2. 执行已经核对过的模型原生推理流程。
3. 将必要的准备步骤和格式转换包含在推理流程内部，用户不需要额外运行 `prepare.py`。
4. 推理结束后直接生成项目规定的标准输出格式。
5. 在原生流程允许时，支持完整数据集运行和受限 smoke/`--max-images` 运行。
6. 将状态、日志和预测结果写入该模型的 `result/` 目录。
7. 发生实际推理错误时返回非零退出码。

模型内部可以使用多个脚本，但对外统一使用以下入口：

```bash
python classification/<contributor>/models/<model-name>/inference.py
python detection/<contributor>/models/<model-name>/inference.py
```

模型专属参数必须记录在该模型的 README 中。从项目根目录和模型目录执行时，默认命令都应当可以工作。

项目中不设置统一的 `run_model.py`、中央模型注册表或所有模型都必须调用的中央适配器。每个模型的推理行为由自己的模型目录负责。只有真正与具体模型无关的任务级评测代码或稳定公共工具，才可以放在模型目录之外。

## 评测入口规范

分类和检测分别使用一个评测入口，两个入口文件名相同：

```text
classification/evaluate.py
detection/evaluate.py
```

模型目录内不放置 `evaluation/` 文件夹，也不设置 `vendor/` 目录。模型只负责生成自己的标准结果，任务目录下的 evaluate.py 读取对应结果并计算指标。

示例：

```bash
python classification/evaluate.py \
  --run-dir classification/<contributor>/models/<model-name>/result/run_001

python detection/evaluate.py \
  --run-dir detection/<contributor>/models/<model-name>/result/run_001
```

检测任务统一使用 detection/evaluate.py 计算 AP。所有需要计算 AP 的模型都必须先把预测结果转换为项目规定的 COCO bbox 格式，然后直接调用 AP 评测逻辑。分类任务统一使用 classification/evaluate.py，检测任务统一使用 detection/evaluate.py。

当前检测模型中，表格结构识别模型 6 和 15 的主要指标是 GriTS。这两个模型只记录 GriTS 评测信息，不纳入统一 AP 批处理。GriTS 所需的单元格文本、行列关系和单元格位置不能由普通 bbox AP 代替。

原本使用 LVIS 数据集或 LVIS 原生评测器的模型，在本项目中也按统一 COCO bbox 输出和 AP 逻辑处理。LVIS 原生评测结果可以作为历史参考，但不再设置单独的 LVIS 评测入口，也不作为本项目的主评测分支。结果协议中应记录实际使用的 GT、类别映射和输入范围。

评测脚本的参数和标准输入输出格式必须写在项目 README 中。评测脚本不能通过模型专属推理分支代替模型推理，也不能要求用户为某个模型单独运行另一套评测入口。

## 路径、结果和可复现性

- 使用 `pathlib` 或同等方式，并以脚本位置为基础解析路径。
- 数据集、权重和结果路径必须可以通过配置或环境变量调整，不应修改源码才能换机器运行。
- 优先使用项目相对路径，并在 README 中说明共享存储根目录。
- 不使用软连接伪造模型目录中的权重、数据或脚本。
- 不得静默复用其他模型的结果目录。
- 每次运行应记录运行命令、环境标识、输入范围和输出位置。
- 环境文件放在项目级或实验级目录，不替代模型目录内必须存在的 `inference.py`、`weights/` 和 `result/`。

## 新增模型检查清单

新增模型前必须确认：

- 模型位于正确的任务目录和贡献者目录下。
- 模型目录包含 `inference.py`、`weights/` 和 `result/`。
- 从项目根目录执行默认推理命令可以运行。
- 准备步骤和格式转换已经包含在 `inference.py` 中。
- 推理直接生成标准输出格式。
- 不依赖个人绝对路径，也不依赖软连接。
- 模型 README 记录权重、数据集、运行命令、预期输出和已知限制。
- 贡献者目录包含 environment/、依赖清单和 model_environment.csv。
- model_environment.csv 明确每个模型使用哪一套环境；存在多套环境时不能遗漏模型映射。
- 对应任务目录的 evaluate.py 可以直接读取模型结果，不需要增加模型专属评测入口。

如果确实需要改变项目结构，必须同时更新本文档和对应任务目录的 README。
