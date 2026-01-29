import asyncio
import sys
import os

# Add current directory to python path
sys.path.append(os.getcwd())

from src.tools.mcp_client import get_mcp_tools_sync, get_mcp_tool_descriptions, get_routing_guide

def test_mcp_loading():
    print("--- Testing MCP Tool Loading ---")
    try:
        tools = get_mcp_tools_sync()
        if not tools:
            print("❌ No MCP tools loaded.")
            return
        
        print(f"✅ Loaded {len(tools)} tools.")
        for t in tools:
            print(f"   - {t.name}")
            
        print("\n--- Testing Description Generation ---")
        desc = get_mcp_tool_descriptions()
        print(desc[:500] + "..." if len(desc) > 500 else desc)

        print("\n--- Testing Routing Guide Generation ---")
        guide = get_routing_guide()
        print(guide)
        
    except Exception as e:
        print(f"❌ Error during MCP loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mcp_loading()
