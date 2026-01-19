import json
import time
import re
from typing import Literal, Optional
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from ..core.state import AgentState
from ..core.llm import get_llm
from ..tools.traffic_tools import traffic_prediction, anomaly_detection, causal_analysis, travel_recommendation
from ..tools.maps import route_planning

# 获取 LLM 实例 (不再 bind_tools)
llm = get_llm()

# 工具映射表
TOOL_MAP = {
    "traffic_prediction": traffic_prediction,
    "anomaly_detection": anomaly_detection,
    "causal_analysis": causal_analysis,
    "travel_recommendation": travel_recommendation,
    "route_planning": route_planning
}

# 手动生成工具描述文档
TOOL_DESC = """
1. route_planning: 路径规划。参数: origin(起点), destination(终点), mode(transit/driving/walking)。
2. traffic_prediction: 预测交通拥堵。参数: origin, destination。
3. anomaly_detection: 检测异常事件。参数: location。
4. causal_analysis: 分析事故影响。参数: affected_area。
5. travel_recommendation: 综合出行推荐。参数: origin, destination。
"""

# 最大重试次数
MAX_RETRY_COUNT = 3

# --- Debug 辅助函数 ---
def _add_debug_log(state: AgentState, log_type: str, content: dict) -> None:
    if "debug_logs" not in state or state["debug_logs"] is None:
        state["debug_logs"] = []
    state["debug_logs"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "type": log_type,
        "content": content
    })

# --- 核心节点逻辑 ---

def perception_node(state: AgentState) -> AgentState:
    """感知节点：提取意图"""
    print("🔍 [Perception] Extracting intent & locations...")
    
    if not state.get("messages"):
        state["messages"] = [HumanMessage(content=state["user_request"])]
    
    user_request = state["user_request"]
    
    # 简单的 JSON 提取 Prompt
    prompt = f"""Extract origin and destination from: "{user_request}".
Return JSON ONLY: {{"origin": "...", "destination": "..."}}. 
If unknown, use empty string."""
    
    try:
        response = llm.invoke(prompt)
        # 暴力清洗
        content = response.content.strip().replace("```json", "").replace("```", "")
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            state["origin"] = data.get("origin", "")
            state["destination"] = data.get("destination", "")
            print(f"   📍 Extracted: {state['origin']} -> {state['destination']}")
    except Exception as e:
        print(f"   ⚠️ Perception failed: {e}")

    state["current_step"] = "perception"
    return state

def planning_node(state: AgentState) -> AgentState:
    """规划节点：手动 Prompt 驱动工具调用"""
    retry_count = state.get("retry_count", 0)
    print(f"📋 [Planning] Reasoning... (attempt {retry_count + 1})")
    
    # ============================================================
    # 核心修改：Construct "Text-to-JSON" Prompt
    # ============================================================
    system_instruction = f"""你是一个交通智能体。
【可用工具】
{TOOL_DESC}

【任务】
请分析用户问题，决定是否需要调用工具。
如果需要，**必须**仅输出一个 JSON 对象，格式如下：
{{
    "tool": "工具名称",
    "args": {{ "参数名": "参数值" }}
}}

**禁止事项**：
1. 不要输出任何Markdown标记（如 ```json）。
2. 不要输出任何解释性文字。
3. 如果用户问路，必须调用 `route_planning`。
4. 如果不需要工具，直接输出 "DIRECT_ANSWER: 你的回答"。
"""
    if state.get("origin"):
        system_instruction += f"\n[已知信息] 起点:{state['origin']} 终点:{state['destination']}"

    # 构建消息 (防止 400 Error)
    messages = list(state.get("messages", []))
    if messages and isinstance(messages[0], HumanMessage):
        if "【可用工具】" not in messages[0].content:
            messages[0] = HumanMessage(content=f"{system_instruction}\n\n用户输入：{messages[0].content}")
    else:
        messages = [HumanMessage(content=f"{system_instruction}\n\n用户输入：{state['user_request']}")]

    # 调用普通 LLM (不带 bind_tools)
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        print(f"   📝 Raw LLM Output: {content[:100]}...")
    except Exception as e:
        print(f"   ❌ LLM Error: {e}")
        response = AIMessage(content="API Error")
        content = ""

    state["messages"].append(response)
    
    # ============================================================
    # 核心修改：手动解析 JSON (Manual Parsing)
    # ============================================================
    outputs = state.get("tool_outputs", [])
    tool_called = False

    # 尝试寻找 JSON 结构
    json_match = re.search(r"\{.*\}", content.replace("\n", ""), re.DOTALL)
    
    if json_match and "DIRECT_ANSWER" not in content:
        try:
            tool_data = json.loads(json_match.group(0))
            tool_name = tool_data.get("tool")
            tool_args = tool_data.get("args", {})
            
            if tool_name in TOOL_MAP:
                print(f"   🛠️ Manually Executing: {tool_name} with {tool_args}")
                tool_func = TOOL_MAP[tool_name]
                
                # 执行工具
                tool_output = tool_func.invoke(tool_args)
                outputs.append({"tool": tool_name, "output": tool_output})
                
                # 伪造一个 ToolMessage (为了保持 State 结构一致性)
                state["messages"].append(ToolMessage(
                    content=str(tool_output),
                    tool_call_id=f"manual_{int(time.time())}", # 伪造 ID
                    name=tool_name
                ))
                tool_called = True
            else:
                print(f"   ⚠️ Unknown tool in JSON: {tool_name}")
        except json.JSONDecodeError:
            print("   ⚠️ JSON Parse Failed")
    
    if not tool_called:
        print("   ⚠️ No valid JSON tool call found.")

    state["tool_outputs"] = outputs
    state["current_step"] = "planning"
    return state

def reflection_node(state: AgentState) -> AgentState:
    """反思节点"""
    print("🤔 [Reflection] Reviewing...")
    retry_count = state.get("retry_count", 0) + 1
    state["retry_count"] = retry_count
    
    # 只要有工具输出，或者重试次数够了，就通过
    if state.get("tool_outputs") or retry_count >= MAX_RETRY_COUNT:
        state["reflection_score"] = 1.0
    else:
        state["reflection_score"] = 0.0
        print("   🛑 No tools used, injecting critique...")
        # 注入更明确的 Prompt
        state["messages"].append(HumanMessage(
            content="Error: You did not output the required JSON tool call. Please output JSON ONLY: {\"tool\": \"route_planning\", \"args\": {...}}"
        ))

    state["current_step"] = "reflection"
    return state

def output_node(state: AgentState) -> AgentState:
    """输出节点"""
    print("✅ [Output] Generating report...")
    context = json.dumps(state.get('tool_outputs', []), ensure_ascii=False)
    prompt = f"根据以下数据回答用户问题（如果是JSON数据请解读它）。\n数据：{context}\n\n用户问题：{state['user_request']}"
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state["recommendation"] = response.content
    state["messages"].append(AIMessage(content=response.content))
    state["current_step"] = "output"
    return state