import gradio as gr
import json
from .agents.traffic_agent import traffic_agent
from .core.state import AgentState

# 全局变量存储最后一次运行的调试日志
_last_debug_logs = []
_last_state = {}

def run_agent_workflow(message, history):
    """
    Interface between Gradio and the LangGraph Agent.
    """
    global _last_debug_logs, _last_state
    
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
    
    # Run the graph
    final_state = traffic_agent.invoke(initial_state)
    
    # 保存调试信息供 Debug 页面使用
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
    
    return final_state["recommendation"]

def get_debug_info():
    """获取调试信息"""
    global _last_debug_logs, _last_state
    
    if not _last_debug_logs:
        return "尚无调试信息。请先在主界面发送一条消息。", "{}"
    
    # 格式化调试日志
    log_lines = []
    for log in _last_debug_logs:
        timestamp = log.get("timestamp", "??:??:??")
        log_type = log.get("type", "unknown")
        content = log.get("content", {})
        
        # 根据类型格式化
        if log_type == "llm_response":
            log_lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            log_lines.append(f"🤖 [{timestamp}] LLM 响应 (耗时: {content.get('elapsed_time', '?')})")
            log_lines.append(f"📝 内容: {content.get('content', '(无)')}")
            if content.get("tool_calls"):
                log_lines.append(f"🛠️ 工具调用:")
                for tc in content["tool_calls"]:
                    log_lines.append(f"   • {tc['name']}: {json.dumps(tc['args'], ensure_ascii=False)}")
        elif log_type == "tool_execution":
            log_lines.append(f"✅ [{timestamp}] 工具执行: {content.get('tool', '?')}")
            log_lines.append(f"   参数: {json.dumps(content.get('args', {}), ensure_ascii=False)}")
            log_lines.append(f"   结果: {json.dumps(content.get('output', {}), ensure_ascii=False, indent=2)}")
        elif log_type == "reflection":
            score = content.get("reflection_score", 0)
            status = "✅ 通过" if score >= 0.6 else "⚠️ 需重试"
            log_lines.append(f"🤔 [{timestamp}] 反思评估: {status}")
            log_lines.append(f"   重试次数: {content.get('retry_count', 0)}, 分数: {score}")
        elif log_type == "perception":
            log_lines.append(f"🔍 [{timestamp}] 感知: {content.get('origin', '?')} → {content.get('destination', '?')}")
        elif log_type == "final_output":
            log_lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            log_lines.append(f"📄 [{timestamp}] 最终报告生成 (耗时: {content.get('elapsed_time', '?')}, 长度: {content.get('report_length', 0)} 字符)")
        elif log_type == "no_tool_calls":
            log_lines.append(f"ℹ️ [{timestamp}] {content.get('note', 'LLM未调用工具')}")
        elif log_type == "tool_error":
            log_lines.append(f"❌ [{timestamp}] 工具错误: {content.get('tool', '?')}")
            log_lines.append(f"   错误: {content.get('error', '未知错误')}")
        else:
            log_lines.append(f"📋 [{timestamp}] {log_type}: {json.dumps(content, ensure_ascii=False)}")
    
    debug_log_text = "\n".join(log_lines) if log_lines else "无调试日志"
    state_json = json.dumps(_last_state, ensure_ascii=False, indent=2)
    
    return debug_log_text, state_json

def create_ui():
    """
    Creates the professional Gradio UI for the Traffic Agent with Debug tab.
    """
    with gr.Blocks(
        title="🚦 智慧交通诱导智能体 (工程化 Demo)"
    ) as demo:
        gr.Markdown("""
        # 🚦 智慧交通诱导智能体
        ### 基于多源异构交通大数据的实时诱导与决策支持系统
        
        本系统集成了博士论文中的核心研究成果：
        *   **时空预测**：基于 Transformer 的路网状态预训练 (Chap 1)
        *   **异常感知**：多模态交通异常检测 (Chap 2)
        *   **因果分析**：GeoDCD 动态因果传播分析 (Chap 3)
        *   **出行推荐**：CDHGNN 对比去偏异构图神经网络 (Chap 4)
        """)
        
        with gr.Tabs():
            # 主聊天界面
            with gr.TabItem("💬 智能对话", id="chat"):
                chatbot = gr.ChatInterface(
                    fn=run_agent_workflow,
                    examples=["从北京南站到清华大学怎么走？", "西二环发生事故，对我去机场有影响吗？", "预测明天周一早高峰的通行状况"],
                    title=None
                )
            
            # Debug 页面
            with gr.TabItem("🐛 调试信息", id="debug"):
                gr.Markdown("### 调试面板\n点击下方按钮刷新，查看最近一次请求的完整 LLM 输出和执行日志。")
                
                refresh_btn = gr.Button("🔄 刷新调试信息", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        debug_logs = gr.Textbox(
                            label="📋 执行日志 (LLM 输出 / 工具调用)",
                            lines=20,
                            max_lines=30,
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        state_info = gr.JSON(
                            label="📊 最终状态摘要"
                        )
                
                refresh_btn.click(
                    fn=get_debug_info,
                    outputs=[debug_logs, state_info]
                )
        
        gr.Markdown("""
        ---
        **技术栈**: LangGraph + OpenAI Gemini-3-Flash + Gaode Maps API + UV Package Manager
        """)
        
    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)

