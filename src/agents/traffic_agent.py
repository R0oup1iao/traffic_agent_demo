from langgraph.graph import StateGraph, START, END
from ..core.state import AgentState
from .nodes import perception_node, planning_node, reflection_node, output_node

def create_agent():
    """
    Constructs the LangGraph StateGraph for the Traffic Guidance Agent.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("perception", perception_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("output", output_node)
    
    # Define Edges
    workflow.add_edge(START, "perception")
    workflow.add_edge("perception", "planning")
    workflow.add_edge("planning", "reflection")
    
    # Conditional Edge: Reflection -> Retry Planning or Success Output
    def should_continue(state: AgentState):
        if state["reflection_score"] < 0.6:
            return "planning"
        return "output"
    
    workflow.add_conditional_edges(
        "reflection",
        should_continue,
        {
            "planning": "planning",
            "output": "output"
        }
    )
    
    workflow.add_edge("output", END)
    
    return workflow.compile()

traffic_agent = create_agent()
传播_agent = traffic_agent # Alias
