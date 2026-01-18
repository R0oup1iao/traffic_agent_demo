# 🚗 交通诱导智能体 Demo

> 博士论文第四章「面向异常态势的交通诱导智能体构建与应用」技术演示

本项目演示了如何使用 **LangGraph + OpenAI + 高德地图 API** 构建一个具备自主决策与编排能力的交通诱导智能体。

---

## 📚 技术栈

| 组件 | 技术 | 作用 | 对应论文章节 |
|:----:|:----:|:----:|:------------:|
| 🧠 认知编排 | [LangGraph](https://github.com/langchain-ai/langgraph) | 实现有向循环图（DCG）认知流 | 6.2.1 节 |
| 🤖 大语言模型 | [OpenAI 兼容 API](https://platform.openai.com/) | 自然语言理解与生成 | 64.2.1 节 |
| 🗺️ 地图服务 | [高德地图 API](https://lbs.amap.com/) | 路径规划、POI 检索 | 6.2.3 节 |
| 🖥️ 交互界面 | [Gradio](https://gradio.app/) | 提供 Web 聊天界面与调试面板 | 6.2.4 节 |

---

## 🚀 快速开始

### 1️⃣ 使用 uv 安装依赖（推荐）

本项目使用 [uv](https://github.com/astral-sh/uv) 作为包管理器：

```powershell
# 安装 uv（如果尚未安装）
pip install uv

# 安装项目依赖
uv sync
```

### 2️⃣ 配置环境变量

创建 `.env` 文件或设置环境变量：

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"   # 必需：OpenAI 兼容 API Key
$env:OPENAI_BASE_URL = "http://127.0.0.1:8045/v1"  # 可选：自定义 API 端点
$env:MODEL_NAME = "gemini-3-flash"            # 可选：模型名称
$env:AMAP_API_KEY = "xxxxxxxxxxxxxxxx"        # 可选：高德地图 API Key
```

> 💡 **提示**：
> - 项目默认配置可在 `src/core/config.py` 中查看和修改
> - 高德地图 API Key 可在 [lbs.amap.com](https://lbs.amap.com/) 免费申请

### 3️⃣ 运行 Gradio 界面

```powershell
# 使用 uv 运行
uv run python -m src.main

# 或激活虚拟环境后运行
.\.venv\Scripts\Activate.ps1
python -m src.main
```

访问 `http://localhost:7860` 即可使用智能体界面。

---

## 📁 项目结构

```
agent_demo/
├── src/                        # 📦 源代码目录
│   ├── main.py                 # 🎯 Gradio 界面入口
│   ├── core/                   # ⚙️ 核心配置
│   │   ├── config.py           #    环境变量与配置管理
│   │   ├── llm.py              #    LLM 客户端初始化
│   │   └── state.py            #    AgentState 状态定义
│   ├── agents/                 # 🤖 智能体模块
│   │   ├── traffic_agent.py    #    LangGraph 图结构定义
│   │   └── nodes.py            #    认知节点实现
│   └── tools/                  # 🛠️ 工具模块
│       ├── traffic_tools.py    #    论文模型工具（预测/感知/因果/推荐）
│       └── maps.py             #    高德地图 API 封装
├── archive/                    # 📂 旧版代码存档
│   ├── smoke_test.py           #    完整版 Smoke Test（FastAPI + MCP）
│   └── simple_agent.py         #    简化版单文件 Agent
├── tests/                      # 🧪 测试目录
├── pyproject.toml              # 📋 项目元数据（uv 管理）
├── requirements.txt            # 📋 传统依赖列表（pip 兼容）
└── README.md                   # 📖 本文档
```

---

## 🏗️ 智能体架构

本 Demo 实现了论文中提出的**有向循环图（DCG）认知流**结构：

```
                    ┌──────────────────────────────────────────────────┐
                    │           交通诱导智能体 (Traffic Guidance Agent) │
                    ├──────────────────────────────────────────────────┤
                    │                                                  │
    用户请求 ──────►│  ┌────────┐   ┌────────┐   ┌────────┐   ┌─────┐ │
                    │  │  感知  │──►│  规划  │──►│  反思  │──►│ 输出 │──► 推荐结果
                    │  │  节点  │   │  节点  │   │  节点  │   │ 节点 │ │
                    │  └────────┘   └────────┘   └───┬────┘   └─────┘ │
                    │                                │                 │
                    │                    ┌───────────┘                 │
                    │                    │ 评分 < 0.6                  │
                    │                    ▼                             │
                    │              ┌──────────┐                        │
                    │              │ 重新规划  │ (循环状态转移)          │
                    │              └──────────┘                        │
                    ├──────────────────────────────────────────────────┤
                    │  【工具层】                                       │
                    │  ├─ 时空预测：路网拥堵预测 (第一章)                │
                    │  ├─ 异常感知：多模态异常检测 (第二章)              │
                    │  ├─ 因果分析：动态因果传播 (第三章)                │
                    │  ├─ 出行推荐：CDHGNN 推荐 (第四章)                │
                    │  └─ 路径规划：高德地图 API                        │
                    └──────────────────────────────────────────────────┘
```

### 核心节点说明

| 节点 | 功能 | 实现文件 |
|:----:|:-----|:--------|
| **感知节点** | 解析用户请求，提取起终点信息 | `src/agents/nodes.py` → `perception_node()` |
| **规划节点** | 调用工具获取交通态势，生成候选方案 | `src/agents/nodes.py` → `planning_node()` |
| **反思节点** | 评估方案可行性，检查安全隐患与效率问题 | `src/agents/nodes.py` → `reflection_node()` |
| **输出节点** | 生成人性化的出行建议报告 | `src/agents/nodes.py` → `output_node()` |

### 状态空间定义

对应论文公式：$S = \{G_{flow}, E_{alert}, M_{context}, P_{candidate}, F_{feedback}, \Lambda\}$

```python
# src/core/state.py
class AgentState(TypedDict):
    user_request: str      # 用户出行请求
    origin: str            # 起点
    destination: str       # 终点
    traffic_status: str    # 当前交通态势 (对应 G_flow, E_alert)
    tool_outputs: list     # 工具调用结果
    candidate_plans: list  # 候选方案 (对应 P_candidate)
    recommendation: str    # 最终推荐
    reflection_score: float # 反思评分 (对应 F_feedback)
    retry_count: int       # 重试次数
    messages: list         # 对话历史 (对应 M_context)
    current_step: str      # 当前步骤 (对应 Λ)
    debug_logs: list       # 调试日志
```

---

## 🛠️ 工具集成

本智能体集成了论文各章节的核心模型作为工具：

| 工具名称 | 论文章节 | 功能描述 |
|:--------|:---------|:---------|
| `traffic_prediction` | 第一章 | 基于 Transformer 的路网时空预测 |
| `anomaly_detection` | 第二章 | 融合 LLM 的多模态异常感知 |
| `causal_analysis` | 第三章 | 基于几何深度学习的动态因果发现 |
| `travel_recommendation` | 第四章 | 对比去偏异构图神经网络（CDHGNN）推荐 |
| `route_planning` | 外部 API | 高德地图路径规划接口 |

> 📝 **注意**：当前工具返回模拟数据用于演示。接入真实模型时，替换 `src/tools/` 中的实现即可。

---

## 🖥️ Gradio 界面

运行后界面包含两个标签页：

### 💬 智能对话

- 自然语言交互，支持出行咨询
- 预设示例问题快速体验
- 实时显示智能体回复

### 🐛 调试信息

- 查看完整的 LLM 输出
- 跟踪工具调用过程
- 查看状态变化和反思评分

---

## 📊 论文对应关系

| Demo 代码 | 论文章节 | 说明 |
|:----------|:---------|:-----|
| `AgentState` 类 | §4.2.1 | 状态空间 $S$ 定义 |
| `perception_node` | §4.2.1 | 感知与丰富节点 |
| `planning_node` | §4.2.1 | 规划决策节点（模拟 CDHGNN） |
| `reflection_node` | §4.2.1 | 自我反思节点 |
| `should_continue` 条件边 | §4.2.1 | 循环状态转移机制 |
| `traffic_tools.py` | §4.2.2 | 论文各章模型工具化 |
| `maps.py` | §4.2.3 | 外部能力接入（高德地图） |

---

## 🔧 扩展开发

### 接入真实的 CDHGNN 模型

```python
# 在 src/tools/traffic_tools.py 中替换模拟逻辑：
from cdhgnn import CDHGNNModel

model = CDHGNNModel.load("path/to/model.pt")

@tool
def travel_recommendation(origin: str, destination: str, user_id: str = "U001"):
    plans = model.recommend(user_id, origin, destination)
    return {"tool": "CDHGNN推荐模型", "result": plans}
```

### 添加新工具

```python
# src/tools/your_tool.py
from langchain_core.tools import tool

@tool
def your_custom_tool(param: str):
    """工具描述，LLM 会根据此描述决定何时调用"""
    # 实现逻辑
    return {"result": "..."}

# 在 src/agents/nodes.py 中注册工具
from ..tools.your_tool import your_custom_tool
tools = [..., your_custom_tool]
```

---

## 📝 运行效果示例

```
============================================================
🚦 智慧交通诱导智能体
============================================================

💬 用户：从北京南站到清华大学怎么走？

🔍 [感知] 解析请求: 北京南站 → 清华大学
🛠️ [规划] 调用工具:
   • traffic_prediction: 获取实时交通状态
   • anomaly_detection: 检测沿途异常
   • travel_recommendation: 生成推荐方案
   • route_planning: 获取详细路线
🤔 [反思] 方案评分: 0.85 ✅
📄 [输出] 生成出行建议报告

💡 推荐结果：
根据当前交通态势分析，建议您乘坐地铁4号线从北京南站
直达五道口站，再步行至清华大学，全程约45分钟，费用5元。
当前该路线拥堵指数较低（0.35），是最优选择。
```

---

## 📂 归档文件说明

`archive/` 目录包含早期版本的代码，仅供参考：

| 文件 | 说明 |
|:-----|:-----|
| `smoke_test.py` | 完整版演示，包含 FastAPI + MCP 协议支持 |
| `simple_agent.py` | 简化版单文件 Agent，使用原生 OpenAI SDK |

---

## 📜 License

本项目为博士论文配套演示代码，仅供学术研究使用。
