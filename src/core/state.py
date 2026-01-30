from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State definition for the Traffic Intelligence Agent.
    """
    # User Input
    user_request: str
    origin: str
    destination: str
    
    # Context & Perception
    traffic_status: str
    tool_outputs: List[dict]
    
    # Decisions & Planning
    candidate_plans: List[dict]
    recommendation: str
    
    # Meta
    reflection_score: float
    retry_count: int
    # 使用 add_messages reducer，消息会自动追加而非覆盖
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    
    # Debug
    debug_logs: List[dict]
