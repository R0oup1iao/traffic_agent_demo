# 🚗 交通诱导智能体 Demo

> 博士论文第四章「面向异常态势的交通诱导智能体构建与应用」技术演示

本项目演示了如何使用 **LangGraph + OpenAI + MCP + FastAPI** 构建一个具备自主决策与编排能力的交通诱导智能体。

---

## 📚 技术栈

| 组件 | 技术 | 作用 | 对应论文章节 |
|:----:|:----:|:----:|:------------:|
| 🧠 认知编排 | [LangGraph](https://github.com/langchain-ai/langgraph) | 实现有向循环图（DCG）认知流 | 4.2.1 节 |
| 🤖 大语言模型 | [OpenAI GPT](https://platform.openai.com/) | 自然语言理解与生成 | 4.2.1 节 |
| 🔌 能力互操作 | [MCP Protocol](https://modelcontextprotocol.io/) | 连接高德地图等外部能力 | 4.2.3 节 |
| 🌐 API 服务 | [FastAPI](https://fastapi.tiangolo.com/) | 提供 RESTful 接口 | 4.2.4 节 |

---

## 🚀 快速开始

### 1️⃣ 创建虚拟环境（推荐）

```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 3️⃣ 配置环境变量

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"   # 必需：OpenAI API Key
$env:AMAP_API_KEY = "xxxxxxxxxxxxxxxx"        # 可选：高德地图开放平台申请
```

> 💡 **提示**：
> - OpenAI API Key 可在 [platform.openai.com](https://platform.openai.com/) 获取
> - 高德地图 API Key 可在 [lbs.amap.com](https://lbs.amap.com/) 免费申请

### 4️⃣ 运行 Smoke Test

```powershell
# 确保已激活虚拟环境
python smoke_test.py
```

---

## 📁 项目结构

```
agent_demo/
├── smoke_test.py       # 🎯 核心演示文件
│   ├── LangGraph 智能体定义
│   ├── MCP Client 示例代码
│   └── FastAPI 接口示例
├── requirements.txt    # 📦 Python 依赖列表
└── README.md           # 📖 本文档
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
                    │  【MCP 层】                                       │
                    │  MCP Client ──► 高德地图 Server ──► 路径规划/POI  │
                    ├──────────────────────────────────────────────────┤
                    │  【API 层】                                       │
                    │  FastAPI ──► /recommend ──► 出行推荐服务          │
                    └──────────────────────────────────────────────────┘
```

### 核心节点说明

| 节点 | 功能 | 实现 |
|:----:|:-----|:-----|
| **感知节点** | 获取当前交通态势、调用 MCP 接口检索动态因果图 | `perception_node()` |
| **规划节点** | 调用 CDHGNN 模型生成候选出行方案 | `planning_node()` |
| **反思节点** | 评估方案可行性，检查安全隐患与效率问题 | `reflection_node()` |
| **输出节点** | 生成人性化的出行建议 | `output_node()` |

### 状态空间定义

对应论文公式：$S = \{G_{flow}, E_{alert}, M_{context}, P_{candidate}, F_{feedback}, \Lambda\}$

```python
class AgentState(TypedDict):
    user_request: str      # 用户出行请求
    origin: str            # 起点
    destination: str       # 终点
    traffic_status: str    # 当前交通态势 (对应 G_flow, E_alert)
    candidate_plans: list  # 候选方案 (对应 P_candidate)
    recommendation: str    # 最终推荐
    reflection_score: float # 反思评分 (对应 F_feedback)
    messages: list         # 对话历史 (对应 M_context)
    current_step: str      # 当前步骤 (对应 Λ)
```

---

## 🔗 MCP 协议说明

**Model Context Protocol (MCP)** 是一种标准化的 AI 能力互操作协议。本项目演示了如何通过 MCP 连接高德地图服务：

```python
# MCP Client 连接高德地图示例
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # 调用路径规划工具
        result = await session.call_tool(
            "route_planning",
            arguments={"origin": "...", "destination": "...", "mode": "transit"}
        )
```

### 可用的高德地图 MCP Server

| 项目 | 地址 |
|:----:|:-----|
| amap-mcp-server | [github.com/sugarforever/amap-mcp-server](https://github.com/sugarforever/amap-mcp-server) |
| gaode | [github.com/perMAIN/gaode](https://github.com/perMAIN/gaode) |

---

## 🌐 FastAPI 接口

运行 FastAPI 服务后，可通过 RESTful API 调用智能体：

```bash
# 启动服务
uvicorn smoke_test:app --host 0.0.0.0 --port 8000
```

### API 端点

| 方法 | 路径 | 说明 |
|:----:|:-----|:-----|
| POST | `/recommend` | 获取出行推荐 |
| GET | `/health` | 健康检查 |

### 请求示例

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "北京西站",
    "destination": "中关村",
    "user_request": "早高峰，推荐最快的方式"
  }'
```

### Swagger UI

访问 `http://localhost:8000/docs` 查看交互式 API 文档。

---

## 📊 论文对应关系

| Demo 代码 | 论文章节 | 说明 |
|:----------|:---------|:-----|
| `AgentState` 类 | §4.2.1 | 状态空间 $S$ 定义 |
| `perception_node` | §4.2.1 | 感知与丰富节点 |
| `planning_node` | §4.2.1 | 规划决策节点（模拟 CDHGNN） |
| `reflection_node` | §4.2.1 | 自我反思节点 |
| `should_retry` 条件边 | §4.2.1 | 循环状态转移机制 |
| `demo_mcp_client` | §4.2.3 | MCP 协议互操作 |
| FastAPI 接口 | §4.2.4 | n8n 执行层（API 化） |

---

## 🔧 扩展开发

### 接入真实的 CDHGNN 模型

```python
# 在 planning_node 中替换模拟逻辑：
from cdhgnn import CDHGNNModel

model = CDHGNNModel.load("path/to/model.pt")
plans = model.recommend(user_id, origin_id, dest_id, timestamp)
```

### 部署高德地图 MCP Server

```bash
# 安装并启动
npx -y @agentic/amap-mcp-server
```

### 添加 n8n 工作流触发

可通过 n8n Webhook 节点触发智能体，实现事件驱动的自动化诱导。

---

## 📝 运行效果示例

```
============================================================
🚗 交通诱导智能体 Smoke Test
============================================================

📦 正在创建 LangGraph 智能体...
✅ 智能体创建成功！

📝 测试用例：
   起点：北京西站
   终点：中关村
   用户需求：我需要从北京西站去中关村，现在是早高峰，请推荐最快的方式

------------------------------------------------------------
🚀 开始运行智能体...
------------------------------------------------------------

🔍 [感知节点] 正在获取交通态势信息...
📋 [规划节点] 正在生成出行方案...
🤔 [反思节点] 正在评估方案可行性...
✅ [输出节点] 正在生成推荐建议...

📊 运行结果
------------------------------------------------------------
🎯 推荐结果：由于早高峰地面交通拥堵，建议您乘坐地铁9号线转10号线前往中关村，预计35分钟到达，避开拥堵路段。
✅ 反思评分：0.87
```

---

## 📜 License

本项目为博士论文配套演示代码，仅供学术研究使用。
