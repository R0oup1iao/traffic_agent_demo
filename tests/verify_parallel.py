import asyncio
import sys
import os
import json

# Add current directory to python path
sys.path.append(os.getcwd())

from src.agents.traffic_agent import traffic_agent
from src.core.state import AgentState
from src.tools.mcp_client import init_mcp_client
from langchain_core.messages import HumanMessage

async def test_parallel_agent():
    print("--- Initializing MCP ---")
    await init_mcp_client()
    
    print("\n--- Running Parallel Agent Test ---")
    initial_state: AgentState = {
        "user_request": "从北京西站到故宫怎么走", # This typically requires two geocodes
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
        final_state = traffic_agent.invoke(initial_state)
        
        print("\n--- Agent Execution Finished ---")
        print(f"Reflection Score: {final_state.get('reflection_score')}")
        
        tool_outputs = final_state.get('tool_outputs', [])
        print(f"Total Tool Outputs: {len(tool_outputs)}")
        
        # Check if we have multiple tool calls in a single planning turn
        # We can check this by seeing if multiple tools appear in the log before reflection?
        # Actually our state just accumulates all outputs.
        # But if the first turn called both geo tools, they should be in the list.
        
        geo_tools = [t for t in tool_outputs if 'geo' in t.get('tool')]
        print(f"Geo tools called: {len(geo_tools)}")
        if len(geo_tools) >= 2:
            print("✅ SUCCESS: Multiple geo tools called (Likely parallel or at least batched in one turn if prompt worked)")
        else:
            print("⚠️ WARNING: Less than 2 geo tools called. Check logs.")

        direction_tools = [t for t in tool_outputs if 'direction' in t.get('tool')]
        print(f"Direction tools called: {len(direction_tools)}")
        
        for output in tool_outputs:
            print(f"Tool: {output.get('tool')}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_parallel_agent())
