"""
高德地图 MCP Client 封装
基于 langchain-mcp-adapters 连接 MCP Server
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from ..core.config import settings

# 全局 MCP Client 实例
_mcp_client: MultiServerMCPClient | None = None
_mcp_tools: list | None = None


async def init_mcp_client():
    """初始化 MCP Client，连接高德地图 MCP Server"""
    global _mcp_client, _mcp_tools
    
    amap_url = f"https://mcp.amap.com/mcp?key={settings.AMAP_API_KEY}"
    print(f"🗺️ Connecting to Amap MCP Server: {amap_url[:50]}...")
    
    _mcp_client = MultiServerMCPClient({
        "amap": {
            "transport": "http",
            "url": amap_url,
        }
    })
    _mcp_tools = await _mcp_client.get_tools()
    print(f"✅ Loaded {len(_mcp_tools)} tools from Amap MCP Server")
    for tool in _mcp_tools:
        print(f"   - {tool.name}: {tool.description[:60] if tool.description else 'No description'}...")
    
    return _mcp_tools


async def get_mcp_tools():
    """异步获取 MCP 工具列表"""
    global _mcp_tools
    if _mcp_tools is None:
        await init_mcp_client()
    return _mcp_tools


def get_mcp_tools_sync():
    """同步获取 MCP 工具（在事件循环中运行）"""
    global _mcp_tools
    if _mcp_tools is not None:
        return _mcp_tools
    
    # 尝试获取当前事件循环
    try:
        loop = asyncio.get_running_loop()
        # 如果已在运行循环中，创建一个 Future
        future = asyncio.ensure_future(init_mcp_client())
        # 这种情况下不能直接等待，需要返回已缓存的工具
        raise RuntimeError("Cannot run sync in async context")
    except RuntimeError:
        # 没有运行中的循环，创建新循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            _mcp_tools = loop.run_until_complete(init_mcp_client())
        finally:
            loop.close()
    
    return _mcp_tools


def get_mcp_tool_map():
    """获取工具名称到工具对象的映射"""
    tools = get_mcp_tools_sync()
    if tools is None:
        return {}
    return {tool.name: tool for tool in tools}


def get_mcp_tool_descriptions():
    """生成 MCP 工具的描述文档"""
    tools = get_mcp_tools_sync()
    if tools is None:
        return "暂无可用的 MCP 工具"
    
    desc_lines = []
    for i, tool in enumerate(tools, 1):
        # 获取工具参数描述
        args_desc = ""
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {}
            props = schema.get('properties', {})
            args = [f"{k}" for k in props.keys()]
            args_desc = f"参数: {', '.join(args)}" if args else ""
        
        desc = tool.description[:100] if tool.description else "无描述"
        desc_lines.append(f"{i}. {tool.name}: {desc}。{args_desc}")
    
    return "\n".join(desc_lines)



