"""
智慧交通诱导智能体 - FastAPI 后端服务
"""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .agents.traffic_agent import traffic_agent
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    处理用户聊天请求，调用 Agent 并返回结果
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
