"""
LangGraph Agent 节点定义
使用 LangGraph 标准模式：bind_tools + ToolNode + add_messages
"""
import json
import time
import re
from typing import Optional, Callable, List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from ..core.state import AgentState
from ..core.llm import get_llm
from ..tools.traffic_tools import traffic_prediction, anomaly_detection, causal_analysis, travel_recommendation
from ..tools.mcp_client import get_mcp_tools_sync

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


# --- Debug 辅助函数 ---
def _add_debug_log(state: AgentState, log_type: str, content: dict) -> None:
    if "debug_logs" not in state or state["debug_logs"] is None:
        state["debug_logs"] = []
    state["debug_logs"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "type": log_type,
        "content": content
    })


# --- 工具获取 ---
# 本地工具列表
LOCAL_TOOLS = [
    traffic_prediction,
    anomaly_detection,
    causal_analysis,
    travel_recommendation,
]


def get_all_tools() -> List:
    """获取所有工具（本地工具 + MCP 工具）"""
    tools = LOCAL_TOOLS.copy()
    try:
        mcp_tools = get_mcp_tools_sync()
        if mcp_tools:
            tools.extend(mcp_tools)
            print(f"✅ Loaded {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"⚠️ Failed to load MCP tools: {e}")
    return tools


# 全局工具列表（延迟初始化）
_all_tools: List | None = None


def _get_tools():
    """获取工具列表（带缓存）"""
    global _all_tools
    if _all_tools is None:
        _all_tools = get_all_tools()
    return _all_tools


# --- 消息预处理 ---
def _normalize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    规范化消息列表：
    1. 将 ToolMessage 的 list content 转换为字符串
    """
    normalized = []
    for msg in messages:
        # 处理 ToolMessage 的 content 为 list 的情况 (MCP 工具返回 TextContent 列表)
        if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
            text_parts = []
            for item in msg.content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    text_parts.append(str(item))
            # 创建新的 ToolMessage
            msg = ToolMessage(
                content='\n'.join(text_parts),
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, 'name', 'unknown')
            )
        normalized.append(msg)
    return normalized


# --- 核心节点逻辑 ---

def perception_node(state: AgentState) -> Dict[str, Any]:
    """感知节点：提取意图，返回更新"""
    print("🔍 [Perception] Extracting intent & locations...")
    _notify_status("perception", "🔍 正在感知用户意图...", "分析您的问题")
    
    # 添加调试日志
    _add_debug_log(state, "perception", {"action": "开始提取用户意图和地点信息"})
    
    user_request = state["user_request"]
    llm = get_llm()
    
    # 提取起点终点
    prompt = f"""Extract origin and destination from: "{user_request}".
Return JSON ONLY: {{"origin": "...", "destination": "..."}}. 
If unknown, use empty string."""
    
    origin = ""
    destination = ""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip().replace("```json", "").replace("```", "")
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            origin = data.get("origin", "")
            destination = data.get("destination", "")
            print(f"   📍 Extracted: {origin} -> {destination}")
            _add_debug_log(state, "perception", {
                "action": "地点提取完成",
                "origin": origin,
                "destination": destination
            })
    except Exception as e:
        print(f"   ⚠️ Perception failed: {e}")
        _add_debug_log(state, "perception", {"action": "地点提取失败", "error": str(e)})

    # 返回状态更新（messages 会自动追加）
    return {
        "origin": origin,
        "destination": destination,
        "current_step": "perception",
        "tool_outputs": [],
        "retry_count": 0,
        "messages": [HumanMessage(content=user_request)],
        "debug_logs": state.get("debug_logs", [])
    }


def call_model(state: AgentState) -> Dict[str, Any]:
    """调用模型节点：使用 bind_tools 让 LLM 决定是否调用工具"""
    retry_count = state.get("retry_count", 0)
    print(f"📋 [Call Model] Reasoning... (attempt {retry_count + 1})")
    _notify_status("planning", "📋 正在规划方案...", f"模型思考中 (尝试 {retry_count + 1})")
    
    _add_debug_log(state, "call_model", {"action": f"开始推理 (第 {retry_count + 1} 次)"})
    
    # 获取工具并绑定到 LLM
    tools = _get_tools()
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    # 构建系统提示
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    
    system_prompt = """你是一个交通智能体助手。请根据用户的问题，使用提供的工具来获取信息并回答。

**重要规则**：
1. 涉及地点查询、路线规划时，必须使用工具获取实时数据
2. 高德地图 API 仅支持经纬度，必须先调用 maps_geo 获取坐标
3. 获取坐标后，再调用 maps_direction_* 获取路线
4. 不要编造数据，必须基于工具返回的真实数据回答"""

    if origin and destination:
        system_prompt += f"\n\n[已知信息] 起点: {origin}, 终点: {destination}"
    
    # 获取消息列表并规范化
    messages = _normalize_messages(list(state.get("messages", [])))
    
    # 在第一条消息前添加系统提示（如果还没有）
    if messages and isinstance(messages[0], HumanMessage):
        if "交通智能体" not in messages[0].content:
            messages[0] = HumanMessage(content=f"{system_prompt}\n\n用户问题：{messages[0].content}")
    
    try:
        # 调用带工具的 LLM
        response = llm_with_tools.invoke(messages)
        
        # 记录响应
        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            print(f"   🛠️ Model requested tools: {tool_names}")
            _add_debug_log(state, "call_model", {
                "action": "LLM 请求工具调用",
                "tools": tool_names
            })
            _notify_status("execution", "⚡ 正在执行工具...", f"调用: {', '.join(tool_names)}")
        else:
            content_preview = response.content[:100] if response.content else "(empty)"
            print(f"   💬 Model response: {content_preview}...")
            _add_debug_log(state, "call_model", {
                "action": "LLM 直接响应",
                "content": content_preview
            })
        
        return {
            "messages": [response],
            "retry_count": retry_count + 1,
            "current_step": "call_model",
            "debug_logs": state.get("debug_logs", [])
        }
        
    except Exception as e:
        print(f"   ❌ LLM Error: {e}")
        _add_debug_log(state, "call_model", {"action": "LLM 错误", "error": str(e)})
        return {
            "messages": [AIMessage(content=f"抱歉，处理请求时出错: {str(e)}")],
            "retry_count": retry_count + 1,
            "current_step": "call_model",
            "debug_logs": state.get("debug_logs", [])
        }


def output_node(state: AgentState) -> Dict[str, Any]:
    """输出节点：生成最终报告"""
    print("✅ [Output] Generating report...")
    _notify_status("output", "📝 正在生成报告...", "整合结果")
    
    _add_debug_log(state, "final_output", {"action": "开始生成最终报告"})
    
    messages = state.get("messages", [])
    
    # 获取最后一条 AI 消息作为响应
    final_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_response = msg.content
            break
    
    # 如果没有直接响应，基于工具输出生成报告
    if not final_response or final_response.startswith("抱歉"):
        # 从消息中收集工具输出
        tool_outputs = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content
                if isinstance(content, list):
                    # 处理 list content
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            tool_outputs.append({"tool": getattr(msg, 'name', 'unknown'), "output": item['text']})
                else:
                    tool_outputs.append({"tool": getattr(msg, 'name', 'unknown'), "output": content})
        
        if tool_outputs:
            llm = get_llm()
            context = json.dumps(tool_outputs, ensure_ascii=False, indent=2)
            prompt = f"""根据以下工具返回的数据，回答用户的问题。请直接给出有用的信息，不要提及"工具"或"API"。

工具数据：
{context}

用户问题：{state['user_request']}

请用中文回答："""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            final_response = response.content
    
    _add_debug_log(state, "final_output", {
        "action": "报告生成完成",
        "length": len(final_response)
    })
    
    return {
        "recommendation": final_response,
        "current_step": "output",
        "debug_logs": state.get("debug_logs", [])
    }


def create_tool_node():
    """创建 ToolNode 实例"""
    tools = _get_tools()
    return ToolNode(tools)


def should_continue(state: AgentState) -> str:
    """判断是否继续调用工具
    
    Returns:
        "tools": 如果需要执行工具
        "output": 如果可以生成最终输出
    """
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    
    # 检查重试次数
    if retry_count >= 5:
        print("   ⚠️ Max retries reached")
        return "output"
    
    # 检查最后一条消息
    if messages:
        last_message = messages[-1]
        
        # 如果是 AIMessage 且有 tool_calls，执行工具
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
    
    return "output"