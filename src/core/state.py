from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

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
    messages: List[BaseMessage]
    current_step: str
    
    # Debug
    debug_logs: List[dict]
