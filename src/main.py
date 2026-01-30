"""
智慧交通诱导智能体 - FastAPI 后端服务
"""
import json
import asyncio
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

from .agents.traffic_agent import traffic_agent
from .agents.nodes import set_status_callback
from .core.state import AgentState


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理，启动时初始化 MCP Client"""
    # Startup
    try:
        from .tools.mcp_client import init_mcp_client
        await init_mcp_client()
        print("✅ MCP Client initialized successfully")
    except Exception as e:
        print(f"⚠️ MCP Client initialization failed: {e}")
        print("   Continuing without MCP tools...")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")


# ===== App 初始化 =====
app = FastAPI(
    title="智慧交通诱导智能体",
    description="基于多源异构交通大数据的实时诱导与决策支持系统",
    version="1.0.0",
    lifespan=lifespan
)

# 静态文件目录
STATIC_DIR = Path(__file__).parent.parent / "static"

# 挂载静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===== 全局变量 =====
_last_debug_logs = []
_last_state = {}

# 当前状态 (用于流式推送)
_current_status = {
    "phase": "idle",
    "text": "",
    "detail": "",
    "updated_at": 0
}
_status_lock = threading.Lock()


def _update_current_status(phase: str, text: str, detail: str = ""):
    """更新当前状态 (线程安全)"""
    global _current_status
    with _status_lock:
        _current_status = {
            "phase": phase,
            "text": text,
            "detail": detail,
            "updated_at": time.time()
        }


# ===== 请求/响应模型 =====
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    success: bool
    recommendation: str = ""
    debug_logs: list = []
    state: dict = {}
    error: str = ""


# ===== 路由 =====

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回主页"""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "Traffic Intelligence Agent"}


@app.get("/api/status")
async def get_current_status():
    """获取当前 Agent 状态 (用于轮询)"""
    with _status_lock:
        return _current_status.copy()


async def generate_stream(message: str) -> AsyncGenerator[str, None]:
    """
    流式生成 Agent 执行过程，通过 SSE 推送状态更新
    使用 astream 实时获取每个节点的执行状态
    """
    global _last_debug_logs, _last_state
    
    # 重置状态
    _update_current_status("perception", "🔍 正在感知用户意图...", "分析您的问题")
    
    # 初始化 Agent 状态
    initial_state: AgentState = {
        "user_request": message,
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
    
    # 发送初始状态
    yield f"data: {json.dumps({'type': 'status', 'phase': 'perception', 'text': '🔍 正在感知用户意图...', 'detail': '分析您的问题'}, ensure_ascii=False)}\n\n"
    
    print(f"\n{'='*50}")
    print(f"📨 收到请求: {message}")
    print(f"{'='*50}")
    
    # 节点名称到状态的映射
    node_status_map = {
        "perception": ("perception", "🔍 正在感知用户意图...", "分析用户问题"),
        "call_model": ("planning", "📋 正在规划方案...", "模型思考中"),
        "tools": ("execution", "⚡ 正在执行工具...", "调用外部服务"),
        "output": ("output", "📝 正在生成报告...", "整合结果"),
    }
    
    final_state = None
    
    try:
        # 使用 astream 流式获取每个节点的执行状态
        async for event in traffic_agent.astream(initial_state, stream_mode="updates"):
            # event 是一个字典，key 是节点名称，value 是该节点返回的状态更新
            for node_name, node_output in event.items():
                print(f"   📌 Node executed: {node_name}")
                
                # 获取对应的状态信息
                if node_name in node_status_map:
                    phase, text, detail = node_status_map[node_name]
                    
                    # 如果是 tools 节点，尝试提取工具调用信息
                    if node_name == "tools":
                        messages = node_output.get("messages", [])
                        tool_names = []
                        for msg in messages:
                            if hasattr(msg, 'name'):
                                tool_names.append(msg.name)
                        if tool_names:
                            detail = f"执行: {', '.join(tool_names)}"
                    
                    # 如果是 call_model 节点，检查是否有工具调用
                    if node_name == "call_model":
                        messages = node_output.get("messages", [])
                        for msg in messages:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                                phase = "execution"
                                text = "🛠️ 正在调用工具..."
                                detail = f"工具: {', '.join(tool_names)}"
                    
                    # 发送状态更新
                    status_data = {
                        'type': 'status',
                        'phase': phase,
                        'text': text,
                        'detail': detail,
                        'node': node_name
                    }
                    yield f"data: {json.dumps(status_data, ensure_ascii=False)}\n\n"
                    _update_current_status(phase, text, detail)
                
                # 保存最新状态
                if node_name == "output":
                    final_state = node_output
        
        # 如果没有从 output 节点获取到状态，尝试获取完整状态
        if final_state is None:
            # 使用 ainvoke 作为后备
            final_state = await traffic_agent.ainvoke(initial_state)
        
        # 合并状态（astream 只返回更新，可能需要合并）
        recommendation = final_state.get("recommendation", "")
        debug_logs = final_state.get("debug_logs", [])
        
        # 保存调试信息
        _last_debug_logs = debug_logs
        _last_state = {
            "user_request": message,
            "origin": final_state.get("origin", ""),
            "destination": final_state.get("destination", ""),
            "traffic_status": final_state.get("traffic_status", ""),
            "retry_count": final_state.get("retry_count", 0),
            "reflection_score": final_state.get("reflection_score", 0.0),
            "tool_outputs_count": len(final_state.get("tool_outputs", [])),
            "current_step": final_state.get("current_step", "")
        }
        
        print(f"✅ 处理完成，生成报告 {len(recommendation)} 字符")
        
        # 发送完成状态
        yield f"data: {json.dumps({'type': 'status', 'phase': 'execution', 'text': '✅ 生成完成', 'detail': '正在输出回复...'}, ensure_ascii=False)}\n\n"
        
        # 发送最终结果
        result_data = {
            'type': 'result',
            'success': True,
            'recommendation': recommendation,
            'debug_logs': _last_debug_logs,
            'state': _last_state
        }
        yield f"data: {json.dumps(result_data, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        print(f"❌ 处理出错: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
    finally:
        _update_current_status("idle", "", "")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式处理用户聊天请求，通过 SSE 实时推送状态
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")
    
    return StreamingResponse(
        generate_stream(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    处理用户聊天请求，调用 Agent 并返回结果（非流式，保持兼容）
    """
    global _last_debug_logs, _last_state
    
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")
    
    try:
        # 初始化 Agent 状态
        initial_state: AgentState = {
            "user_request": message,
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
        
        # 运行 Agent 图
        print(f"\n{'='*50}")
        print(f"📨 收到请求: {message}")
        print(f"{'='*50}")
        
        final_state = await traffic_agent.ainvoke(initial_state)
        
        # 保存调试信息
        _last_debug_logs = final_state.get("debug_logs", [])
        _last_state = {
            "user_request": final_state.get("user_request", ""),
            "origin": final_state.get("origin", ""),
            "destination": final_state.get("destination", ""),
            "traffic_status": final_state.get("traffic_status", ""),
            "retry_count": final_state.get("retry_count", 0),
            "reflection_score": final_state.get("reflection_score", 0.0),
            "tool_outputs_count": len(final_state.get("tool_outputs", [])),
            "current_step": final_state.get("current_step", "")
        }
        
        print(f"✅ 处理完成，生成报告 {len(final_state['recommendation'])} 字符")
        
        return ChatResponse(
            success=True,
            recommendation=final_state["recommendation"],
            debug_logs=_last_debug_logs,
            state=_last_state
        )
        
    except Exception as e:
        print(f"❌ 处理出错: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            success=False,
            error=str(e)
        )


@app.get("/api/debug")
async def get_debug_info():
    """获取最后一次运行的调试信息"""
    global _last_debug_logs, _last_state
    
    return {
        "debug_logs": _last_debug_logs,
        "state": _last_state
    }


# ===== 启动入口 =====
def main():
    """启动服务"""
    import uvicorn
    print("\n" + "="*60)
    print("🚦 智慧交通诱导智能体 - 服务启动中...")
    print("="*60)
    print("📍 访问地址: http://localhost:8000")
    print("📖 API 文档: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
