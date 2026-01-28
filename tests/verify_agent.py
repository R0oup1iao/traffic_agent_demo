import asyncio
import sys
import os
import json

# Add current directory to python path
sys.path.append(os.getcwd())

from src.agents.traffic_agent import traffic_agent
from src.core.state import AgentState
from src.tools.mcp_client import init_mcp_client

async def test_agent():
    print("--- Initializing MCP ---")
    await init_mcp_client()
    
    print("\n--- Running Agent Test ---")
    initial_state: AgentState = {
        "user_request": "从北京西站到故宫怎么走",
        "origin": "",
        "destination": "",
        "traffic_status": "",
        "tool_outputs": [],
        "candidate_plans": [],
        "recommendation": "",
        "reflection_score": 0.0,
        "retry_count": 0,
        "messages": [],
        "current_step": "init",
        "debug_logs": []
    }
    
    try:
        # Use invoke directly (synchronous call wrapping async tools inside)
        # Note: traffic_agent is a compiled LangGraph, invoke is sync if the graph is sync.
        # But our tools are async. Let's see how LangGraph handles it. 
        # Actually our mcp_client.py handles async execution within sync tools if needed, 
        # or we rely on LangGraph's async support.
        # Let's try synchronous invoke first as that's what main.py uses.
        
        final_state = traffic_agent.invoke(initial_state)
        
        print("\n--- Agent Execution Finished ---")
        print(f"Reflection Score: {final_state.get('reflection_score')}")
        print(f"Tool Outputs Count: {len(final_state.get('tool_outputs', []))}")
        
        for output in final_state.get('tool_outputs', []):
            print(f"Tool: {output.get('tool')}")
            # print(f"Output: {str(output.get('output'))[:100]}...")

        print("\n--- Recommendation ---")
        # print(final_state.get("recommendation"))
        
        # Check if maps tools were used
        tools_used = [o.get('tool') for o in final_state.get('tool_outputs', [])]
        if any('maps_' in t for t in tools_used):
            print("\n✅ SUCCESS: MCP Tools were used!")
        else:
            print("\n❌ FAILURE: MCP Tools were NOT used.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Since our tools might need an event loop, we should run this in a way that supports it?
    # Actually main.py uses a thread pool to run traffic_agent.invoke.
    # But for a simple script, we can just run it.
    # However, init_mcp_client is async.
    asyncio.run(test_agent())
