"""
智慧交通诱导智能体 - FastAPI 后端服务
"""
import json
import asyncio
import time
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

from .agents.traffic_agent import traffic_agent
from .agents.nodes import set_status_callback
from .core.state import AgentState

# ===== App 初始化 =====
app = FastAPI(
    title="智慧交通诱导智能体",
    description="基于多源异构交通大数据的实时诱导与决策支持系统",
    version="1.0.0"
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
    yield f"data: {json.dumps({'type': 'status', 'phase': 'perception', 'text': '🔍 正在感知用户意图...', 'detail': '分析您的问题'})}\n\n"
    
    print(f"\n{'='*50}")
    print(f"📨 收到请求: {message}")
    print(f"{'='*50}")
    
    # 设置状态回调
    last_sent_status = {"phase": "perception", "text": "", "detail": ""}
    
    def on_status_change(phase: str, text: str, detail: str = ""):
        _update_current_status(phase, text, detail)
    
    set_status_callback(on_status_change)
    
    try:
        # 在后台线程运行 Agent
        import concurrent.futures
        result_holder = {"final_state": None, "error": None}
        
        def run_agent():
            try:
                result_holder["final_state"] = traffic_agent.invoke(initial_state)
            except Exception as e:
                result_holder["error"] = str(e)
                import traceback
                traceback.print_exc()
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_agent)
        
        # 轮询状态更新并推送
        while not future.done():
            await asyncio.sleep(0.2)  # 200ms 轮询间隔
            
            with _status_lock:
                current = _current_status.copy()
            
            # 只有状态变化时才推送
            if (current["phase"] != last_sent_status["phase"] or 
                current["text"] != last_sent_status["text"]):
                yield f"data: {json.dumps({'type': 'status', 'phase': current['phase'], 'text': current['text'], 'detail': current['detail']})}\n\n"
                last_sent_status = current.copy()
        
        # 等待完成
        future.result()
        executor.shutdown(wait=False)
        
        # 检查是否有错误
        if result_holder["error"]:
            yield f"data: {json.dumps({'type': 'error', 'error': result_holder['error']})}\n\n"
            return
        
        final_state = result_holder["final_state"]
        
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
        
        # 发送完成状态
        yield f"data: {json.dumps({'type': 'status', 'phase': 'execution', 'text': '✅ 生成完成', 'detail': '正在输出回复...'})}\n\n"
        
        # 发送最终结果
        yield f"data: {json.dumps({'type': 'result', 'success': True, 'recommendation': final_state['recommendation'], 'debug_logs': _last_debug_logs, 'state': _last_state})}\n\n"
        
    except Exception as e:
        print(f"❌ 处理出错: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    finally:
        # 清除回调
        set_status_callback(None)
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
        
        final_state = traffic_agent.invoke(initial_state)
        
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
