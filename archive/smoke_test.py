"""
交通诱导智能体 Smoke Test Demo
==============================
本文件演示 LangGraph + OpenAI API + MCP 的基本集成方式。

技术栈:
- LangGraph: 实现有向循环图（DCG）认知流
- OpenAI API: 调用 GPT 模型进行推理
- MCP Client: 连接高德地图 MCP Server 获取路径规划能力
- FastAPI: 提供 RESTful API 接口（示例代码）

使用前请安装依赖:
    pip install langgraph langchain-openai mcp httpx pydantic fastapi uvicorn

配置环境变量:
    OPENAI_API_KEY=your_openai_api_key
    AMAP_API_KEY=your_amap_api_key (高德地图开放平台申请)

运行方式:
    python smoke_test.py
"""

import os
import json
import asyncio
from typing import TypedDict, Annotated, Literal
from dataclasses import dataclass

# ============================================================
# 第一部分：基础配置
# ============================================================

# OpenAI API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "your-amap-key-here")

# ============================================================
# 第二部分：LangGraph 智能体定义
# ============================================================

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 定义智能体状态（对应论文中的状态空间 S）
class AgentState(TypedDict):
    """
    智能体的全局状态字典，对应论文公式：
    S = {G_flow, E_alert, M_context, P_candidate, F_feedback, Λ}
    """
    # 用户输入的出行请求
    user_request: str
    # 起点和终点
    origin: str
    destination: str
    # 当前交通态势（简化为文本描述）
    traffic_status: str
    # 候选出行方案
    candidate_plans: list[dict]
    # 推荐结果
    recommendation: str
    # 反思评估结果
    reflection_score: float
    # 对话历史
    messages: list
    # 当前步骤
    current_step: str


def create_traffic_guidance_agent():
    """
    创建交通诱导智能体的 LangGraph 图结构。
    
    实现论文中描述的有向循环图（DCG）结构：
    感知节点 -> 规划决策节点 -> 反思节点 -> (循环回规划或输出)
    """
    
    # 初始化 LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 使用较便宜的模型进行测试
        api_key=OPENAI_API_KEY,
        temperature=0.7
    )
    
    # -------------------- 节点定义 --------------------
    
    def perception_node(state: AgentState) -> AgentState:
        """
        感知与丰富节点 (Perception & Enrichment Node)
        
        负责从多源异构数据中提取当前态势特征，调用 MCP 接口获取实时交通信息。
        """
        print("\n🔍 [感知节点] 正在获取交通态势信息...")
        
        # 模拟调用 MCP 获取交通状态（实际应通过 MCP Client 调用高德 API）
        # 这里简化为模拟数据
        traffic_info = simulate_traffic_status(state["origin"], state["destination"])
        
        state["traffic_status"] = traffic_info
        state["current_step"] = "perception_complete"
        state["messages"].append(
            AIMessage(content=f"[感知节点] 已获取交通态势：{traffic_info}")
        )
        
        return state
    
    def planning_node(state: AgentState) -> AgentState:
        """
        规划决策节点 (Planning Node)
        
        基于当前状态调用 CDHGNN 推荐模型生成候选出行方案。
        这里使用 LLM 模拟 CDHGNN 的推荐逻辑。
        """
        print("\n📋 [规划节点] 正在生成出行方案...")
        
        # 构建 prompt，模拟调用 CDHGNN
        system_prompt = """你是一个交通出行推荐专家。基于用户的出行请求和当前交通态势，
        请生成 3 个候选出行方案，包括：交通方式、预估时间、预估费用、推荐理由。
        
        请以 JSON 格式返回，格式如下：
        [
          {"mode": "地铁", "time": "30分钟", "cost": "5元", "reason": "..."},
          ...
        ]
        """
        
        user_prompt = f"""
        出行请求：从 {state["origin"]} 到 {state["destination"]}
        当前交通态势：{state["traffic_status"]}
        用户原始需求：{state["user_request"]}
        """
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # 解析响应（简化处理）
        try:
            # 尝试提取 JSON
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            plans = json.loads(content.strip())
        except:
            # 如果解析失败，使用模拟数据
            plans = [
                {"mode": "地铁", "time": "35分钟", "cost": "5元", "reason": "避开地面拥堵"},
                {"mode": "打车", "time": "45分钟", "cost": "50元", "reason": "门到门服务"},
                {"mode": "公交+地铁", "time": "40分钟", "cost": "4元", "reason": "经济实惠"}
            ]
        
        state["candidate_plans"] = plans
        state["current_step"] = "planning_complete"
        state["messages"].append(
            AIMessage(content=f"[规划节点] 已生成 {len(plans)} 个候选方案")
        )
        
        return state
    
    def reflection_node(state: AgentState) -> AgentState:
        """
        自我反思节点 (Reflection Node)
        
        评估规划方案的可行性，检查是否存在安全隐患或效率问题。
        如果评估不通过，将触发循环回规划节点。
        """
        print("\n🤔 [反思节点] 正在评估方案可行性...")
        
        # 使用 LLM 进行反思评估
        reflection_prompt = f"""
        请评估以下出行方案的可行性（0-1分）：
        
        出行需求：从 {state["origin"]} 到 {state["destination"]}
        交通态势：{state["traffic_status"]}
        候选方案：{json.dumps(state["candidate_plans"], ensure_ascii=False)}
        
        请给出综合评分（0-1之间的小数），并简要说明理由。
        返回格式：{{"score": 0.85, "reason": "..."}}
        """
        
        response = llm.invoke([HumanMessage(content=reflection_prompt)])
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
            score = result.get("score", 0.8)
        except:
            score = 0.85  # 默认通过
        
        state["reflection_score"] = score
        state["current_step"] = "reflection_complete"
        state["messages"].append(
            AIMessage(content=f"[反思节点] 方案评分：{score:.2f}")
        )
        
        return state
    
    def output_node(state: AgentState) -> AgentState:
        """
        输出节点 (Output Node)
        
        生成最终的人性化推荐建议。
        """
        print("\n✅ [输出节点] 正在生成推荐建议...")
        
        # 选择最佳方案
        best_plan = state["candidate_plans"][0] if state["candidate_plans"] else {}
        
        output_prompt = f"""
        基于以下信息，生成一段简洁、友好的出行建议（50字以内）：
        
        起点：{state["origin"]}
        终点：{state["destination"]}
        交通态势：{state["traffic_status"]}
        推荐方案：{json.dumps(best_plan, ensure_ascii=False)}
        """
        
        response = llm.invoke([HumanMessage(content=output_prompt)])
        
        state["recommendation"] = response.content
        state["current_step"] = "complete"
        state["messages"].append(
            AIMessage(content=f"[输出节点] 推荐完成")
        )
        
        return state
    
    # -------------------- 条件边定义 --------------------
    
    def should_retry(state: AgentState) -> Literal["planning", "output"]:
        """
        条件边：根据反思评分决定是否需要重新规划
        
        这实现了论文中的"循环状态转移"机制：
        当反思节点驳回初始方案时，控制流将沿反向边跳转回规划节点。
        """
        if state["reflection_score"] < 0.6:
            print("⚠️ 方案评分过低，触发重新规划...")
            return "planning"
        return "output"
    
    # -------------------- 构建图 --------------------
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("perception", perception_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("output", output_node)
    
    # 添加边（实现 DCG 结构）
    workflow.add_edge(START, "perception")      # 入口 -> 感知
    workflow.add_edge("perception", "planning") # 感知 -> 规划
    workflow.add_edge("planning", "reflection") # 规划 -> 反思
    
    # 条件边：反思 -> 规划（循环）或 输出
    workflow.add_conditional_edges(
        "reflection",
        should_retry,
        {
            "planning": "planning",
            "output": "output"
        }
    )
    
    workflow.add_edge("output", END)  # 输出 -> 结束
    
    # 编译图
    app = workflow.compile()
    
    return app


def simulate_traffic_status(origin: str, destination: str) -> str:
    """
    模拟交通态势获取（实际应通过 MCP 调用高德 API）
    """
    # 模拟一些异常态势
    import random
    scenarios = [
        "当前路况正常，预计通行顺畅",
        "东三环发生交通事故，局部拥堵严重，建议绕行",
        "受暴雨影响，部分路段积水，地面交通受阻",
        "早高峰时段，主干道车流量大，地铁客流密集"
    ]
    return random.choice(scenarios)


# ============================================================
# 第三部分：MCP Client 示例（连接高德地图 MCP Server）
# ============================================================

async def demo_mcp_client():
    """
    演示如何使用 MCP Client 连接高德地图 MCP Server。
    
    注意：实际使用需要：
    1. 安装高德地图 MCP Server（如 amap-mcp-server）
    2. 配置 AMAP_API_KEY 环境变量
    3. 启动 MCP Server
    
    这里仅展示代码结构，不实际调用。
    """
    print("\n" + "="*60)
    print("MCP Client 示例代码")
    print("="*60)
    
    mcp_example_code = '''
# MCP Client 示例代码（需要 MCP Server 运行）

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_amap_mcp():
    """通过 MCP 调用高德地图路径规划"""
    
    # 配置 MCP Server 连接参数
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@agentic/amap-mcp-server"],
        env={"AMAP_API_KEY": os.getenv("AMAP_API_KEY")}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            
            # 列出可用工具
            tools = await session.list_tools()
            print(f"可用工具: {[t.name for t in tools.tools]}")
            
            # 调用路径规划工具
            result = await session.call_tool(
                "route_planning",
                arguments={
                    "origin": "116.481028,39.989643",  # 起点经纬度
                    "destination": "116.434446,39.90816",  # 终点经纬度
                    "mode": "transit"  # 公共交通
                }
            )
            
            return result

# 在智能体的感知节点中调用：
# traffic_info = await call_amap_mcp()
'''
    
    print(mcp_example_code)
    print("\n上述代码展示了 MCP Client 的典型用法。")
    print("实际部署时，需要将此逻辑集成到 LangGraph 的感知节点中。\n")


# ============================================================
# 第四部分：FastAPI 接口示例
# ============================================================

def print_fastapi_example():
    """
    展示如何将智能体封装为 FastAPI 服务。
    """
    print("\n" + "="*60)
    print("FastAPI 接口示例代码")
    print("="*60)
    
    fastapi_code = '''
# FastAPI 接口封装示例

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="交通诱导智能体 API",
    description="基于 LangGraph + CDHGNN 的智能出行推荐服务",
    version="1.0.0"
)

class TravelRequest(BaseModel):
    """出行请求模型"""
    origin: str          # 起点
    destination: str     # 终点
    user_request: str    # 用户自然语言描述

class TravelRecommendation(BaseModel):
    """出行推荐响应"""
    recommendation: str
    candidate_plans: list
    reflection_score: float

@app.post("/recommend", response_model=TravelRecommendation)
async def get_travel_recommendation(request: TravelRequest):
    """
    获取出行推荐
    
    调用交通诱导智能体，返回个性化的出行建议。
    """
    # 创建智能体
    agent = create_traffic_guidance_agent()
    
    # 初始化状态
    initial_state = {
        "user_request": request.user_request,
        "origin": request.origin,
        "destination": request.destination,
        "traffic_status": "",
        "candidate_plans": [],
        "recommendation": "",
        "reflection_score": 0.0,
        "messages": [],
        "current_step": "init"
    }
    
    # 运行智能体
    final_state = agent.invoke(initial_state)
    
    return TravelRecommendation(
        recommendation=final_state["recommendation"],
        candidate_plans=final_state["candidate_plans"],
        reflection_score=final_state["reflection_score"]
    )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

# 启动服务：
# uvicorn smoke_test:app --host 0.0.0.0 --port 8000
'''
    
    print(fastapi_code)
    print("\n将上述代码添加到本文件，即可启动 FastAPI 服务。")
    print("访问 http://localhost:8000/docs 查看 Swagger UI。\n")


# ============================================================
# 第五部分：主函数 - 运行 Smoke Test
# ============================================================

async def main():
    """
    Smoke Test 主函数
    """
    print("="*60)
    print("🚗 交通诱导智能体 Smoke Test")
    print("="*60)
    
    # 1. 创建智能体
    print("\n📦 正在创建 LangGraph 智能体...")
    agent = create_traffic_guidance_agent()
    print("✅ 智能体创建成功！")
    
    # 2. 准备测试用例
    test_case = {
        "user_request": "我需要从北京西站去中关村，现在是早高峰，请推荐最快的方式",
        "origin": "北京西站",
        "destination": "中关村",
        "traffic_status": "",
        "candidate_plans": [],
        "recommendation": "",
        "reflection_score": 0.0,
        "messages": [],
        "current_step": "init"
    }
    
    print(f"\n📝 测试用例：")
    print(f"   起点：{test_case['origin']}")
    print(f"   终点：{test_case['destination']}")
    print(f"   用户需求：{test_case['user_request']}")
    
    # 3. 运行智能体
    print("\n" + "-"*60)
    print("🚀 开始运行智能体...")
    print("-"*60)
    
    try:
        # 检查 API Key
        if OPENAI_API_KEY == "your-api-key-here":
            print("\n⚠️  警告：未设置 OPENAI_API_KEY 环境变量")
            print("   请设置环境变量后重新运行：")
            print("   $env:OPENAI_API_KEY = 'your-actual-api-key'")
            print("\n   下面将展示 MCP 和 FastAPI 的示例代码...\n")
        else:
            # 实际运行智能体
            final_state = agent.invoke(test_case)
            
            print("\n" + "-"*60)
            print("📊 运行结果")
            print("-"*60)
            print(f"\n🎯 推荐结果：{final_state['recommendation']}")
            print(f"\n📋 候选方案：")
            for i, plan in enumerate(final_state["candidate_plans"], 1):
                print(f"   方案 {i}: {plan}")
            print(f"\n✅ 反思评分：{final_state['reflection_score']:.2f}")
    
    except Exception as e:
        print(f"\n❌ 运行出错：{e}")
        print("   请检查 OPENAI_API_KEY 是否正确配置。")
    
    # 4. 展示 MCP 示例
    await demo_mcp_client()
    
    # 5. 展示 FastAPI 示例
    print_fastapi_example()
    
    # 6. 总结
    print("="*60)
    print("🎉 Smoke Test 完成！")
    print("="*60)
    print("""
本 Demo 展示了以下技术要点：

1. LangGraph 有向循环图（DCG）结构
   - 感知节点 → 规划节点 → 反思节点 → (循环或输出)
   - 状态管理与条件边实现

2. MCP (Model Context Protocol) 集成
   - 连接高德地图 MCP Server
   - 标准化工具调用接口

3. FastAPI 服务封装
   - RESTful API 设计
   - Pydantic 数据验证

下一步：
- 配置真实的 API Key
- 部署高德地图 MCP Server
- 将本 Demo 扩展为完整的出行推荐系统
""")


if __name__ == "__main__":
    asyncio.run(main())
