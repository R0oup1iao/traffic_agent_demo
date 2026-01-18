import json
import time
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from ..core.state import AgentState
from ..core.llm import get_llm
from ..tools.traffic_tools import traffic_prediction, anomaly_detection, causal_analysis, travel_recommendation
from ..tools.maps import route_planning

llm = get_llm()
tools = [traffic_prediction, anomaly_detection, causal_analysis, travel_recommendation, route_planning]
llm_with_tools = llm.bind_tools(tools)

# 最大重试次数
MAX_RETRY_COUNT = 2

def _add_debug_log(state: AgentState, log_type: str, content: dict) -> None:
    """添加调试日志到 state"""
    if "debug_logs" not in state or state["debug_logs"] is None:
        state["debug_logs"] = []
    state["debug_logs"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "type": log_type,
        "content": content
    })

def perception_node(state: AgentState) -> AgentState:
    """感知节点：获取交通态势"""
    print("🔍 [Perception] Detecting traffic status...")
    
    # 初始化 debug_logs
    if "debug_logs" not in state or state["debug_logs"] is None:
        state["debug_logs"] = []
    
    # 这里模拟一次简单的感知，如果用户没提供起点终点，先尝试提取
    if not state.get("origin") or not state.get("destination"):
        # 简单从消息中提取（实际可以用LLM辅助）
        msg = state["user_request"]
        # 简单逻辑演示
        if "到" in msg:
            parts = msg.split("到")
            state["origin"] = parts[0].replace("从", "").strip()
            state["destination"] = parts[1].split("，")[0].split(" ")[0].strip()
    
    _add_debug_log(state, "perception", {
        "origin": state.get("origin", "未知"),
        "destination": state.get("destination", "未知")
    })
    
    # 调用异常检测
    res = anomaly_detection.invoke({"location": state.get("origin", "未知区域")})
    state["traffic_status"] = str(res)
    state["messages"].append(AIMessage(content=f"已完成初步交通态势感知：{state['traffic_status']}"))
    state["current_step"] = "perception"
    
    _add_debug_log(state, "perception_result", {
        "traffic_status": state["traffic_status"]
    })
    
    return state

def planning_node(state: AgentState) -> AgentState:
    """规划节点：调用工具并生成候选方案"""
    retry_count = state.get("retry_count", 0)
    print(f"📋 [Planning] Generating plans... (attempt {retry_count + 1}/{MAX_RETRY_COUNT + 1})")
    
    # 包含感知信息和历史消息
    messages = [
        SystemMessage(content=f"""你是一个交通专家。当前交通态势为：{state.get('traffic_status', '未知')}。
请根据用户的需求，调用合适的工具（预测、异常、因果、路径规划、推荐）来生成最佳建议。
重要提示：请尽量一次性调用所有需要的工具，避免多次反复调用。"""),
        HumanMessage(content=state["user_request"])
    ]
    # 包含之前的工具输出和模型思考（如果有）
    if state.get("messages"):
        messages.extend(state["messages"])
    
    # 调用 LLM
    start_time = time.time()
    response = llm_with_tools.invoke(messages)
    elapsed_time = time.time() - start_time
    
    # 记录 LLM 完整输出到 debug_logs
    _add_debug_log(state, "llm_response", {
        "content": response.content if response.content else "(无文本内容)",
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"]} 
            for tc in (response.tool_calls or [])
        ],
        "elapsed_time": f"{elapsed_time:.2f}s"
    })
    
    print(f"   📝 LLM Response: {response.content[:100] if response.content else '(工具调用)'}...")
    
    state["messages"].append(response)
    
    # 执行工具调用
    outputs = state.get("tool_outputs", [])
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"   🛠️ Calling tool: {tool_name} with {tool_args}")
            
            try:
                if tool_name == "traffic_prediction":
                    tool_output = traffic_prediction.invoke(tool_args)
                elif tool_name == "anomaly_detection":
                    tool_output = anomaly_detection.invoke(tool_args)
                elif tool_name == "causal_analysis":
                    tool_output = causal_analysis.invoke(tool_args)
                elif tool_name == "travel_recommendation":
                    tool_output = travel_recommendation.invoke(tool_args)
                elif tool_name == "route_planning":
                    tool_output = route_planning.invoke(tool_args)
                else:
                    tool_output = {"error": f"Unknown tool: {tool_name}"}
                
                outputs.append({"tool": tool_name, "output": tool_output})
                
                # 记录工具调用结果
                _add_debug_log(state, "tool_execution", {
                    "tool": tool_name,
                    "args": tool_args,
                    "output": tool_output
                })
                
                # 将工具结果作为 ToolMessage 回传给对话流，防止 LLM 重复调用
                state["messages"].append(ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                ))
            except Exception as e:
                print(f"   ❌ Tool execution failed: {e}")
                _add_debug_log(state, "tool_error", {
                    "tool": tool_name,
                    "error": str(e)
                })
    else:
        print("   ℹ️ No tool calls in response")
        _add_debug_log(state, "no_tool_calls", {
            "note": "LLM 未返回工具调用，可能已有足够信息"
        })
    
    state["tool_outputs"] = outputs
    state["current_step"] = "planning"
    return state

def reflection_node(state: AgentState) -> AgentState:
    """反思节点：评估方案"""
    print("🤔 [Reflection] Evaluating plans...")
    
    # 增加重试计数
    retry_count = state.get("retry_count", 0) + 1
    state["retry_count"] = retry_count
    
    # 评估逻辑优化：
    # 1. 如果有工具产出，直接通过
    # 2. 如果重试次数达到上限，也通过（避免无限循环）
    # 3. 如果 LLM 有响应内容但没有工具调用，说明已有足够信息，也通过
    
    has_tool_outputs = bool(state.get("tool_outputs"))
    max_retries_reached = retry_count >= MAX_RETRY_COUNT
    
    # 检查最后一条消息是否是有内容的 AI 响应
    last_ai_message = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
            last_ai_message = msg
            break
        elif isinstance(msg, AIMessage) and msg.content:
            last_ai_message = msg
            break
    
    has_meaningful_response = last_ai_message and last_ai_message.content
    
    if has_tool_outputs or max_retries_reached or has_meaningful_response:
        state["reflection_score"] = 0.9
        reason = []
        if has_tool_outputs:
            reason.append("有工具输出")
        if has_meaningful_response:
            reason.append("有LLM响应")
        if max_retries_reached:
            reason.append("达到重试上限")
        print(f"   ✅ Passed: {', '.join(reason)}")
    else:
        state["reflection_score"] = 0.5
        print(f"   ⚠️ Retry needed (attempt {retry_count}/{MAX_RETRY_COUNT})")
    
    _add_debug_log(state, "reflection", {
        "retry_count": retry_count,
        "reflection_score": state["reflection_score"],
        "has_tool_outputs": has_tool_outputs,
        "max_retries_reached": max_retries_reached
    })
    
    state["current_step"] = "reflection"
    return state

def output_node(state: AgentState) -> AgentState:
    """输出节点：生成报告"""
    print("✅ [Output] Generating final report...")
    
    context = f"用户需求: {state['user_request']}\n态势: {state['traffic_status']}\n工具结果: {json.dumps(state['tool_outputs'], ensure_ascii=False)}"
    
    prompt = f"请根据以下背景信息，为用户生成一个专业、友好且详尽的交通诱导报告。包含预测、异常、因果和具体建议。\n\n{context}"
    
    start_time = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    elapsed_time = time.time() - start_time
    
    state["recommendation"] = response.content
    state["messages"].append(AIMessage(content="报告生成完毕。"))
    state["current_step"] = "output"
    
    _add_debug_log(state, "final_output", {
        "report_length": len(response.content),
        "elapsed_time": f"{elapsed_time:.2f}s"
    })
    
    return state
