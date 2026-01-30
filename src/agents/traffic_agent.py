"""
Traffic Intelligence Agent - LangGraph 图定义
使用标准 ToolNode + bind_tools 模式
"""
from langgraph.graph import StateGraph, START, END
from ..core.state import AgentState
from .nodes import (
    perception_node,
    call_model,
    output_node,
    create_tool_node,
    should_continue,
)


def create_agent():
    """
    构建 LangGraph StateGraph 实现的交通智能体。
    
    图结构:
        START → perception → call_model ⟷ tools → output → END
    """
    workflow = StateGraph(AgentState)
    
    # 创建 ToolNode
    tool_node = create_tool_node()
    
    # 添加节点
    workflow.add_node("perception", perception_node)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("output", output_node)
    
    # 定义边
    workflow.add_edge(START, "perception")
    workflow.add_edge("perception", "call_model")
    
    # 条件边：call_model -> tools 或 output
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tools": "tools",
            "output": "output"
        }
    )
    
    # 工具执行后返回 call_model（继续对话）
    workflow.add_edge("tools", "call_model")
    
    # 最终输出
    workflow.add_edge("output", END)
    
    return workflow.compile()


# 创建全局 agent 实例
traffic_agent = create_agent()
