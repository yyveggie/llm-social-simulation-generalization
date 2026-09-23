# 实验运行说明书

本说明书用于运行已经拆分好的 Python 实验脚本。现在推荐使用 `.py` 脚本运行实验，不需要打开 `.ipynb` notebook。

## 1. 基本运行流程

### 1.1 进入项目根目录

在终端中确保当前位置是项目根目录

### 1.2 配置 LLM 服务商和 API Key

运行实验主要改两个地方：

- `configs/llm_configs.yaml`：配置服务商、模型名、API 地址和 API Key
- `configs/run_config.yaml`：选择当前使用哪个服务商配置

先在 `configs/llm_configs.yaml` 中确认或新增服务商配置。本地运行时，可以直接把自己的 API Key 填到 `api_key`：

```yaml
llms:
  - name: moonshot
    base_url: https://api.moonshot.cn/v1
    api_key: 你的真实 API Key
    model: kimi-k2-turbo-preview
    max_concurrency: 100
    timeout: 60
```

然后在 `configs/run_config.yaml` 中选择当前服务商：

```yaml
active_llm: moonshot
```

也可以不改 `run_config.yaml`，直接在运行命令里用 `--llm` 临时切换模型配置：

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --llm deepseek --experiment dictator
```

如果要新增服务商，请在 `configs/llm_configs.yaml` 里添加一项：

```yaml
- name: my_provider
  base_url: https://你的服务商兼容接口/v1
  api_key: 你的真实 API Key
  model: your-model-name
  timeout: 60
```

然后运行：

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --llm my_provider --experiment dictator
```

## 2. 按配置运行所有启用实验

运行 `configs/run_config.yaml` 中所有 `enabled: true` 的实验：

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml
```

## 3. 单独运行一个实验

单独运行一个实验时使用：

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment 实验名
```

注意：只要使用 `--experiment 实验名`，即使该实验在 `configs/run_config.yaml` 中是 `enabled: false`，也会运行。

如果要临时切换服务商/模型配置，可以加 `--llm`：

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --llm openai --experiment dictator
python scripts/run_configured_experiments.py --config configs/run_config.yaml --llm deepseek --experiment dictator
python scripts/run_configured_experiments.py --config configs/run_config.yaml --llm qwen --experiment dictator
```

如果还想临时指定输出目录，可以加 `--results-dir`：

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --llm deepseek --results-dir records_new/deepseek_test --experiment dictator
```

## 4. 实验命令总表

### 4.1 Big Five 大五人格测试

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment bigfive
```

读取 `data/bigfive.tsv`，让模型逐题回答大五人格问卷。

### 4.2 Dictator 基础独裁者博弈

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment dictator
```

模型决定如何在自己和另一名玩家之间分配 100 美元。

### 4.3 Dictator explained 要求解释版

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment dictator_explained
```

模型先给出分配金额，然后解释自己的选择。

### 4.4 Dictator w_ex 旧命名兼容版

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment dictator_w_ex
```

和 `dictator_explained` 使用相同提示词。保留该名称是为了对应旧日志中的 `w_ex`。

### 4.5 Dictator wo_ex 旧命名兼容版

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment dictator_wo_ex
```

和基础 `dictator` 使用相同提示词。保留该名称是为了对应旧日志中的 `wo_ex`。

### 4.6 Dictator witnessed 见证语境版

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment dictator_witnessed
```

提示词中加入 `The game host hands you $100`，模拟主持人/实验者见证分配过程的语境。

### 4.7 Dictator paired female 对方为女性玩家

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment dictator_paired_female
```

对方玩家被描述为女性，用于观察性别描述是否影响分配。

### 4.8 Dictator paired male 对方为男性玩家

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment dictator_paired_male
```

对方玩家被描述为男性，用于观察性别描述是否影响分配。

### 4.9 Dictator occupations described 职业设定版

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment dictator_occupations_described
```

分别让模型扮演不同职业，再运行基础独裁者博弈。

当前包含职业：

- `mathematician`
- `public relations specialist`
- `journalist`
- `investment fund manager`
- `economist`
- `legislator`

注意：这个实验调用量较大，大约是 `职业数 × n_instances × 2 轮对话`。

### 4.10 Ultimatum proposer 最后通牒博弈-提议者

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_proposer
```

模型作为 Proposer，决定给 Responder 分多少钱。

### 4.11 Ultimatum proposer paired female

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_proposer_paired_female
```

模型作为 Proposer，对方 Responder 被描述为女性玩家。

### 4.12 Ultimatum proposer paired male

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_proposer_paired_male
```

模型作为 Proposer，对方 Responder 被描述为男性玩家。

### 4.13 Ultimatum proposer occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_proposer_occupations_described
```

分别让模型扮演不同职业，再运行最后通牒博弈提议者实验。

### 4.14 Ultimatum responder 最后通牒博弈-回应者

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_responder
```

模型作为 Responder，回答自己最低接受多少钱。

### 4.15 Ultimatum responder paired female

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_responder_paired_female
```

模型作为 Responder，对方 Proposer 被描述为女性玩家。

### 4.16 Ultimatum responder paired male

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_responder_paired_male
```

模型作为 Responder，对方 Proposer 被描述为男性玩家。

### 4.17 Ultimatum responder paired chatbot

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_responder_paired_chatbot
```

模型作为 Responder，对方 Proposer 被描述为 chatbot。

### 4.18 Ultimatum responder occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment ultimatum_responder_occupations_described
```

分别让模型扮演不同职业，再运行最后通牒博弈回应者实验。

### 4.19 Trust investor 信任博弈-投资者

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_investor
```

模型作为 Investor，决定从 100 美元中投资多少给 Banker。

### 4.20 Trust investor occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_investor_occupations_described
```

分别让模型扮演不同职业，再运行信任博弈投资者实验。

### 4.21 Trust banker 10 信任博弈-银行家，投资额 10

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_10
```

假设 Investor 投资 10 美元，金额变为 30 美元，模型作为 Banker 决定返还多少。

### 4.22 Trust banker 10 occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_10_occupations_described
```

分别让模型扮演不同职业，再运行投资额为 10 美元的信任博弈银行家实验。

### 4.23 Trust banker 50 信任博弈-银行家，投资额 50

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_50
```

假设 Investor 投资 50 美元，金额变为 150 美元，模型作为 Banker 决定返还多少。

### 4.24 Trust banker 50 occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_50_occupations_described
```

分别让模型扮演不同职业，再运行投资额为 50 美元的信任博弈银行家实验。

### 4.25 Trust banker 100 信任博弈-银行家，投资额 100

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_100
```

假设 Investor 投资 100 美元，金额变为 300 美元，模型作为 Banker 决定返还多少。

### 4.26 Trust banker 100 occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_100_occupations_described
```

分别让模型扮演不同职业，再运行投资额为 100 美元的信任博弈银行家实验。

### 4.27 Bomb risk 炸弹风险游戏

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment bomb_risk
```

模型在 100 个盒子中选择打开多少个，在收益和踩雷风险之间权衡。

相关配置在 `configs/run_config.yaml` 中：

```yaml
bomb_risk:
  enabled: false
  n_instances: 80
  only_first: false
```

其中：

- `only_first: true`：只运行第一轮选择
- `only_first: false`：运行多轮反馈实验

### 4.28 Holt-Laury 风险偏好测试

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment holt_laury
```

模型连续回答 10 组 A/B 彩票选择题，用于测量风险偏好。A 通常是较安全选项，B 通常是较高风险高收益选项。

### 4.29 Beauty Contest 选美/猜数博弈

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment beauty_contest
```

模型与 4 个玩家一起选择 0-100 的数字，目标是最接近所有玩家平均数的 2/3。实验包含多轮反馈路径。

### 4.30 Public goods 公共品博弈

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment public_goods
```

模型每轮决定从 20 美元中贡献多少给公共项目，并回答收益计算。

相关配置在 `configs/run_config.yaml` 中：

```yaml
public_goods:
  enabled: false
  n_instances: 30
  explicit: false
  accept_wrong_payoff: true
```

其中：

- `explicit`：是否在收益问题中显式提供其他玩家平均收益
- `accept_wrong_payoff`：模型算错收益时是否仍保留该条记录

### 4.31 Prisoner's dilemma 囚徒困境 / Push-Pull 卡牌博弈

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment prisoners_dilemma
```

模型在多轮中选择 `Push` 或 `Pull`，用于观察合作/背叛行为。

### 4.32 其他已实现的实验变体命令

下面这些命令是本仓库模块化 runner 中已经实现的可运行入口，主要用于复现原 notebook 中的变体、补充解释型提示、职业角色设定或不同回合设定。这里的实验名是代码入口名称，不代表原论文中的正式命名。

#### 4.32.1 Trust investor explained

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_investor_explained
```

信任博弈投资者版本。模型作为 Investor 决定从 100 美元中投资多少给 Banker，并要求模型解释自己的投资理由。

#### 4.32.2 Trust banker explained

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_10_explained
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_50_explained
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment trust_banker_100_explained
```

信任博弈 Banker 版本。模型分别面对 Investor 投入 10、50、100 美元的情形，决定返还多少，并解释自己的返还理由。

#### 4.32.3 Public Goods loss

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment public_goods_loss
```

公共品博弈的 `loss` 命名兼容入口。该入口使用配置中的 `other_contributions: [0, 0]`，对应旧记录中其他玩家总贡献较低的运行设定。

#### 4.32.4 Public Goods occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment public_goods_occupations_described
```

公共品博弈职业角色设定版。模型会在不同职业身份的 system message 下运行，用于比较职业描述对贡献行为的影响。

#### 4.32.5 Public Goods loss occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment public_goods_loss_occupations_described
```

公共品博弈 `loss` 设定和职业角色设定的组合版本。既使用低其他玩家贡献设定，也在不同职业身份下运行。

#### 4.32.6 Prisoner's Dilemma five rounds pull

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment prisoners_dilemma_five_rounds_pull
```

Push-Pull 卡牌博弈五轮版本。对手行为序列为前两轮 `Pull`、后两轮 `Push`，观察模型在多轮反馈后的选择变化。

#### 4.32.7 Prisoner's Dilemma two rounds push

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment prisoners_dilemma_two_rounds_push
```

Push-Pull 卡牌博弈两轮版本。第一轮后给定对手选择 `Push`，观察模型第二轮如何回应。

#### 4.32.8 Prisoner's Dilemma two rounds pull

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment prisoners_dilemma_two_rounds_pull
```

Push-Pull 卡牌博弈两轮版本。第一轮后给定对手选择 `Pull`，观察模型第二轮如何回应。

#### 4.32.9 Prisoner's Dilemma occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment prisoners_dilemma_occupations_described
```

Push-Pull 卡牌博弈职业角色设定版。模型会在不同职业身份的 system message 下运行，用于比较职业描述对合作/背叛行为的影响。

#### 4.32.10 Prisoner's Dilemma five rounds pull occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment prisoners_dilemma_five_rounds_pull_occupations_described
```

Push-Pull 卡牌博弈五轮版本与职业角色设定的组合版本。对手行为序列为前两轮 `Pull`、后两轮 `Push`，并在不同职业身份下运行。

#### 4.32.11 Prisoner's Dilemma two rounds push occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment prisoners_dilemma_two_rounds_push_occupations_described
```

Push-Pull 卡牌博弈两轮 `Push` 反馈版本与职业角色设定的组合版本。

#### 4.32.12 Prisoner's Dilemma two rounds pull occupations described

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --experiment prisoners_dilemma_two_rounds_pull_occupations_described
```

Push-Pull 卡牌博弈两轮 `Pull` 反馈版本与职业角色设定的组合版本。

## 5. 输出文件位置

默认输出目录由 `configs/run_config.yaml` 中的 `results_dir` 控制：

```yaml
results_dir: null  # null 时自动保存到 records_new/<active_llm>
```

输出文件名格式类似：

```text
records_new/deepseek/dictator_deepseek_2026_05_04-07_50_00_PM.json
```

每运行一次，都会生成一个新的带时间戳 JSON 文件。

如果运行时传入 `--results-dir`，则优先使用命令行指定的目录：

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml --llm qwen --results-dir records_new/qwen_round1 --experiment dictator
```

## 6. 常用配置项

配置文件位置：

```text
configs/run_config.yaml
```

关键配置：

```yaml
active_llm: moonshot
llm_config_path: configs/llm_configs.yaml
results_dir: null

run:
  save_metadata: true
  print_except: true
  default_n_instances: 30
  max_retries: 3
```

含义：

- `active_llm`：当前使用的模型配置名称，对应 `configs/llm_configs.yaml`
- `llm_config_path`：LLM 服务商配置文件路径
- `results_dir`：结果保存目录；为 `null` 时自动使用 `records_new/<active_llm>`
- `save_metadata`：是否在结果 JSON 中保存模型和实验配置
- `print_except`：失败时是否打印错误
- `default_n_instances`：默认重复次数
- `max_retries`：单个样本失败时最多重试次数，防止无限卡住

命令行参数会覆盖配置文件：

- `--llm deepseek`：临时使用 `configs/llm_configs.yaml` 中名为 `deepseek` 的配置
- `--results-dir records_new/test`：临时指定输出目录

## 7. 如何默认启用多个实验

如果想一次运行多个实验，可以在 `configs/run_config.yaml` 中把对应实验改为：

```yaml
enabled: true
```

然后运行：

```bash
python scripts/run_configured_experiments.py --config configs/run_config.yaml
```

## 8. 内置 LLM 配置示例

`configs/llm_configs.yaml` 已经提供了几个 OpenAI-compatible 服务商示例：

- `moonshot`
- `openai`
- `deepseek`
- `qwen`
- `siliconflow`

这些配置的 `api_key` 字段可以直接填写对应服务商的 API Key。公开上传前请确认没有提交真实 Key：

```yaml
llms:
  - name: deepseek
    base_url: https://api.deepseek.com/v1
    api_key: 你的真实 API Key
    model: deepseek-chat
    timeout: 60
```

如果你要接入其他服务商，只需要复制一项并修改：

- `name`：你运行时传给 `--llm` 的名称
- `base_url`：服务商的 OpenAI-compatible API 地址
- `api_key`：服务商 API Key
- `model`：服务商支持的模型名
- `timeout`：单次请求超时时间

如果某个服务商不是 OpenAI-compatible 格式，就需要改 `experiments/common.py` 里的 `ChatClient`。

## 9. 推荐调试方式

第一次跑新实验时，建议先把 `n_instances` 改小，例如：

```yaml
n_instances: 3
```

确认能正常生成结果后，再改回：

```yaml
n_instances: 30
```

## 10. 目前主要源码文件

- `scripts/run_configured_experiments.py`：统一入口脚本
- `configs/run_config.yaml`：运行控制配置
- `configs/llm_configs.yaml`：模型/API 配置
- `experiments/common.py`：通用工具、LLM client、保存结果、重试逻辑
- `experiments/bigfive.py`：大五人格实验
- `experiments/economic_games.py`：Dictator、Ultimatum、Trust 等经济博弈
- `experiments/risk_games.py`：Bomb Risk、Holt-Laury、Beauty Contest
- `experiments/public_goods.py`：Public Goods
- `experiments/prisoners_dilemma.py`：Prisoner's Dilemma
