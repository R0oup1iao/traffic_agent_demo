import json
import time
import re
import asyncio
from typing import Literal, Optional, Callable
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from ..core.state import AgentState
from ..core.llm import get_llm
from ..tools.traffic_tools import traffic_prediction, anomaly_detection, causal_analysis, travel_recommendation
from ..tools.mcp_client import get_mcp_tool_map, get_mcp_tool_descriptions

# 获取 LLM 实例 (不再 bind_tools)
llm = get_llm()

# 全局状态回调函数
_status_callback: Optional[Callable[[str, str, str], None]] = None

def set_status_callback(callback: Optional[Callable[[str, str, str], None]]):
    """设置状态回调函数"""
    global _status_callback
    _status_callback = callback

def _notify_status(phase: str, text: str, detail: str = ""):
    """通知状态变化"""
    if _status_callback:
        try:
            _status_callback(phase, text, detail)
        except Exception as e:
            print(f"Status callback error: {e}")

# 本地工具映射表（不含 MCP 工具）
LOCAL_TOOL_MAP = {
    "traffic_prediction": traffic_prediction,
    "anomaly_detection": anomaly_detection,
    "causal_analysis": causal_analysis,
    "travel_recommendation": travel_recommendation,
}

# 本地工具描述
LOCAL_TOOL_DESC = """
1. traffic_prediction: 预测交通拥堵。参数: origin, destination。
2. anomaly_detection: 检测异常事件。参数: location。
3. causal_analysis: 分析事故影响。参数: affected_area。
4. travel_recommendation: 综合出行推荐。参数: origin, destination。
"""

def get_tool_map():
    """获取完整工具映射（本地工具 + MCP工具）"""
    tool_map = LOCAL_TOOL_MAP.copy()
    try:
        mcp_tools = get_mcp_tool_map()
        tool_map.update(mcp_tools)
    except Exception as e:
        print(f"⚠️ Failed to load MCP tools: {e}")
    return tool_map

def get_tool_desc():
    """获取完整工具描述（本地工具 + MCP工具）"""
    desc = LOCAL_TOOL_DESC
    try:
        mcp_desc = get_mcp_tool_descriptions()
        if mcp_desc:
            desc += "\n【高德地图MCP工具】\n" + mcp_desc
    except Exception as e:
        print(f"⚠️ Failed to get MCP tool descriptions: {e}")
    return desc

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
    _notify_status("perception", "🔍 正在感知用户意图...", "分析您的问题")
    
    # 添加调试日志
    _add_debug_log(state, "perception", {"action": "开始提取用户意图和地点信息"})
    
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
            _add_debug_log(state, "perception", {
                "action": "地点提取完成",
                "origin": state["origin"],
                "destination": state["destination"]
            })
    except Exception as e:
        print(f"   ⚠️ Perception failed: {e}")
        _add_debug_log(state, "perception", {"action": "地点提取失败", "error": str(e)})

    state["current_step"] = "perception"
    return state

def planning_node(state: AgentState) -> AgentState:
    """规划节点：手动 Prompt 驱动工具调用"""
    retry_count = state.get("retry_count", 0)
    print(f"📋 [Planning] Reasoning... (attempt {retry_count + 1})")
    _notify_status("planning", "📋 正在规划方案...", f"模型思考中 (尝试 {retry_count + 1})")
    
    _add_debug_log(state, "llm_response", {"action": f"开始规划 (第 {retry_count + 1} 次)"})
    
    # ============================================================
    # 核心修改：Construct "Text-to-JSON" Prompt
    # ============================================================
    tool_desc = get_tool_desc()  # 动态获取工具描述
    
    system_instruction = f"""你是一个交通智能体。
【可用工具】
{tool_desc}

【任务】
请分析用户问题，决定是否需要调用工具。
如果需要，**必须**输出一个 JSON 列表，格式如下：
[
  {{ "tool": "工具名称", "args": {{ "参数名": "参数值" }} }},
  {{ "tool": "工具名称", "args": {{ "参数名": "参数值" }} }}
]

**禁止事项**：
1. 不要输出任何Markdown标记（如 ```json）。
2. 不要输出任何解释性文字。
3. 工具名必须从【可用工具】列表中选择，不要编造工具名。
4. 如果不需要工具，直接输出 "DIRECT_ANSWER: 你的回答"。

**重要提示**：
- 只要涉及地点查询、路线规划、距离测量，**必须**使用工具，严禁编造数据。
- 高德地图API仅支持经纬度作为起终点，因此**必须先调用 maps_geo 获取经纬度**。
- 支持一次性调用多个工具（例如同时获取起点和终点的坐标）。
- 即使你知道大概路线，也必须调用工具获取实时准确信息。
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
        _add_debug_log(state, "llm_response", {"action": "LLM 响应", "content": content[:200]})
    except Exception as e:
        print(f"   ❌ LLM Error: {e}")
        response = AIMessage(content="API Error")
        content = ""
        _add_debug_log(state, "llm_response", {"action": "LLM 错误", "error": str(e)})

    state["messages"].append(response)
    
    # ============================================================
    # 核心修改：手动解析 JSON List (Manual Parsing)
    # ============================================================
    outputs = state.get("tool_outputs", [])
    tool_calls = []

    # 尝试寻找 JSON 结构 (List or Object)
    # 匹配方括号 [...] 或 花括号 {...}
    json_match = re.search(r"(\[.*\]|\{.*\})", content.replace("\n", ""), re.DOTALL)
    
    if json_match and "DIRECT_ANSWER" not in content:
        try:
            parsed_data = json.loads(json_match.group(0))
            if isinstance(parsed_data, dict):
                tool_calls.append(parsed_data)
            elif isinstance(parsed_data, list):
                tool_calls.extend(parsed_data)
            
            if tool_calls:
                print(f"   🛠️ Scheduled {len(tool_calls)} tools for execution")
                _notify_status("execution", "⚡ 正在执行工具...", f"并发调用 {len(tool_calls)} 个工具")
                
                tool_map = get_tool_map()
                
                # 并发执行工具
                import concurrent.futures
                import random
                
                def execute_tool(tool_call):
                    t_name = tool_call.get("tool")
                    t_args = tool_call.get("args", {})
                    
                    if t_name not in tool_map:
                        return {"tool": t_name, "error": f"Unknown tool: {t_name}"}
                    
                    try:
                        t_func = tool_map[t_name]
                        
                        # 检查是否是异步函数
                        if asyncio.iscoroutinefunction(t_func.ainvoke):
                            # 异步工具需要特殊处理
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # 如果事件循环正在运行，使用 ThreadPoolExecutor 运行异步任务
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(asyncio.run, t_func.ainvoke(t_args))
                                        result = future.result(timeout=30) # 增加超时
                                else:
                                    # 如果事件循环未运行，直接运行
                                    result = loop.run_until_complete(t_func.ainvoke(t_args))
                            except RuntimeError:
                                # 如果在另一个线程中调用，且没有事件循环，则创建并运行
                                result = asyncio.run(t_func.ainvoke(t_args))
                        else:
                            # 同步工具直接调用 invoke
                            result = t_func.invoke(t_args)
                            
                        return {"tool": t_name, "output": result}
                        
                    except Exception as e:
                        return {"tool": t_name, "error": str(e)}

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    results = list(executor.map(execute_tool, tool_calls))
                
                for res in results:
                    outputs.append(res)
                    # Log result
                    if "error" in res:
                        print(f"     ❌ {res['tool']} failed: {res['error']}")
                        _add_debug_log(state, "tool_error", {"tool": res['tool'], "error": res['error']})
                    else:
                        print(f"     ✅ {res['tool']} completed")
                        _add_debug_log(state, "tool_execution", {"tool": res['tool'], "status": "完成"})
                        
                    # 伪造 ToolMessage
                    state["messages"].append(ToolMessage(
                        content=str(res.get("output", res.get("error"))),
                        tool_call_id=f"manual_{int(time.time())}_{random.randint(0,1000)}",
                        name=res["tool"]
                    ))

            else:
                print("   ⚠️ Empty JSON list found")

        except json.JSONDecodeError as e:
            print("   ⚠️ JSON Parse Failed")
            _add_debug_log(state, "tool_error", {"error": f"JSON 解析失败: {str(e)}"})
    else:
        print("   ⚠️ No valid JSON tool call found.")

    state["tool_outputs"] = outputs
    state["current_step"] = "planning"
    return state

def reflection_node(state: AgentState) -> AgentState:
    """反思节点 - 智能判断任务是否完成"""
    print("🤔 [Reflection] Reviewing...")
    _notify_status("reflection", "🤔 正在反思评估...", "检查结果质量")
    
    retry_count = state.get("retry_count", 0) + 1
    state["retry_count"] = retry_count
    
    tool_outputs = state.get("tool_outputs", [])
    user_request = state.get("user_request", "")
    
    _add_debug_log(state, "reflection", {
        "action": "评估结果",
        "retry_count": retry_count,
        "tool_outputs_count": len(tool_outputs)
    })
    
    # 判断是否是路线规划问题
    is_routing_question = any(kw in user_request for kw in ["怎么走", "怎么去", "路线", "导航", "到达", "前往"])
    
    # 检查是否调用了路线规划工具
    called_tools = [o.get("tool", "") for o in tool_outputs]
    has_direction_tool = any("direction" in t for t in called_tools)
    has_geo_tool = any("geo" in t for t in called_tools)
    
    # 判断任务是否完成
    task_complete = False
    
    if retry_count >= MAX_RETRY_COUNT:
        # 重试次数用尽，强制通过
        task_complete = True
        print(f"   ⚠️ Max retries reached ({MAX_RETRY_COUNT})")
    elif is_routing_question:
        # 路线规划问题：需要调用 maps_direction_* 工具
        if has_direction_tool:
            task_complete = True
            print("   ✅ Route planning completed with direction tool")
        elif has_geo_tool and not has_direction_tool:
            # 只调用了地理编码，还需要继续调用路线规划
            task_complete = False
            print("   🔄 Got coordinates, need to call direction tool next")
        else:
            task_complete = False
    elif tool_outputs:
        # 非路线规划问题：有工具输出就通过
        task_complete = True
    
    if task_complete:
        state["reflection_score"] = 1.0
        _add_debug_log(state, "reflection", {"result": "通过", "score": 1.0})
    else:
        state["reflection_score"] = 0.0
        _add_debug_log(state, "reflection", {"result": "需要继续", "score": 0.0})
        
        # 注入提示，引导下一步
        if is_routing_question and has_geo_tool and not has_direction_tool:
            # 从 geo 结果中提取坐标
            geo_results = [o for o in tool_outputs if "geo" in o.get("tool", "")]
            coords_list = []
            for geo_result in geo_results:
                output = geo_result.get("output", "")
                # 尝试从输出中提取坐标
                if isinstance(output, str):
                    # 匹配 location 字段的坐标
                    loc_match = re.search(r'"location":\s*"([0-9.]+,[0-9.]+)"', output)
                    if loc_match:
                        coords_list.append(loc_match.group(1))
                elif isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and 'text' in item:
                            loc_match = re.search(r'"location":\s*"([0-9.]+,[0-9.]+)"', item['text'])
                            if loc_match:
                                coords_list.append(loc_match.group(1))
            
            origin = state.get("origin", "")
            destination = state.get("destination", "")
            
            if len(coords_list) >= 2:
                # 已有两个坐标，可以调用路线规划
                state["messages"].append(HumanMessage(
                    content=f"已获取起点坐标 {coords_list[0]} 和终点坐标 {coords_list[1]}。请调用 maps_direction_transit_integrated 工具，参数: origin=\"{coords_list[0]}\", destination=\"{coords_list[1]}\"。输出 JSON 格式。"
                ))
            elif len(coords_list) == 1:
                # 只有一个坐标，需要获取另一个
                missing_place = destination if origin else origin
                state["messages"].append(HumanMessage(
                    content=f"已获取坐标 {coords_list[0]}。还需要获取 \"{destination}\" 的坐标。请调用 maps_geo，参数 address=\"{destination}\"。"
                ))
            else:
                # 没有提取到坐标，让 LLM 先获取起点坐标
                state["messages"].append(HumanMessage(
                    content=f"请先用 maps_geo 获取起点 \"{origin}\" 的坐标，参数 address=\"{origin}\"。"
                ))
        else:

            # 没有任何工具调用，但却是路线问题
            if is_routing_question:
                state["messages"].append(HumanMessage(
                    content="错误：回答路线问题**必须**使用工具。请先调用 `maps_geo` 获取起点或终点的经纬度。禁止直接回答。"
                ))
            else:
                state["messages"].append(HumanMessage(
                    content="Error: 请从【可用工具】列表中选择合适的工具。格式: {\"tool\": \"工具名\", \"args\": {...}}"
                ))

    state["current_step"] = "reflection"
    return state

def output_node(state: AgentState) -> AgentState:
    """输出节点"""
    print("✅ [Output] Generating report...")
    _notify_status("execution", "📝 正在生成报告...", "整合结果")
    
    _add_debug_log(state, "final_output", {"action": "开始生成最终报告"})
    
    context = json.dumps(state.get('tool_outputs', []), ensure_ascii=False)
    prompt = f"根据以下数据回答用户问题（如果是JSON数据请解读它）。\n数据：{context}\n\n用户问题：{state['user_request']}"
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state["recommendation"] = response.content
    state["messages"].append(AIMessage(content=response.content))
    state["current_step"] = "output"
    
    _add_debug_log(state, "final_output", {
        "action": "报告生成完成",
        "length": len(response.content)
    })
    
    return state