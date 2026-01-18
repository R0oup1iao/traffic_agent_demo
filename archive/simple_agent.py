"""
交通诱导智能体 - 简单版本 (带 Gradio 对话界面)
=================================================

🎯 目标：快速跑通流程，截图放论文

运行方式：
    python simple_agent.py

然后打开浏览器访问 http://localhost:7860
"""

import os
import json
import gradio as gr
from openai import OpenAI

# ============================================================
# 配置
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-0d450dd391d3431d895d24dbde5d7a46")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8045/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash")

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
) if OPENAI_API_KEY else None

# ============================================================
# 模拟的"工具"（你论文中的模型）
# ============================================================

def tool_traffic_prediction(origin: str, destination: str, time: str) -> dict:
    """
    🔧 工具1：时空预测模型（第一章）
    模拟调用你论文第一章的 Transformer 时空预测模型
    """
    return {
        "tool": "时空预测模型",
        "source": "第一章：基于Transformer的路网时空预训练",
        "result": {
            "拥堵指数": 0.72,
            "预测速度": "35 km/h",
            "置信度": 0.89,
            "备注": f"预测 {time} 从 {origin} 到 {destination} 的交通状态"
        }
    }

def tool_anomaly_detection(location: str) -> dict:
    """
    🔧 工具2：异常感知模型（第二章）
    模拟调用你论文第二章的多模态异常感知模型
    """
    # 模拟一些异常场景
    anomalies = [
        {"类型": "交通事故", "位置": "东三环", "影响时长": "约2小时", "严重程度": "中等"},
        {"类型": "道路施工", "位置": "西直门桥", "影响时长": "持续至本周五", "严重程度": "轻微"},
        {"类型": "无异常", "位置": location, "影响时长": "-", "严重程度": "-"}
    ]
    import random
    result = random.choice(anomalies)
    
    return {
        "tool": "异常感知模型",
        "source": "第二章：融合LLM的多模态异常感知",
        "result": result
    }

def tool_causal_analysis(affected_area: str) -> dict:
    """
    🔧 工具3：因果分析模型（第三章）
    模拟调用你论文第三章的 GeoDCD 因果发现框架
    """
    return {
        "tool": "因果分析模型",
        "source": "第三章：基于几何深度学习的动态因果发现",
        "result": {
            "影响传播路径": f"{affected_area} → 二环辅路 → 西直门",
            "预计波及时间": "30-45分钟",
            "因果强度": 0.78,
            "建议绕行": "北三环或四环"
        }
    }

def tool_travel_recommendation(user_id: str, origin: str, destination: str) -> dict:
    """
    🔧 工具4：出行推荐模型 CDHGNN（第四章）
    模拟调用你论文第四章的对比去偏异构图神经网络
    """
    recommendations = [
        {"方式": "地铁", "时间": "35分钟", "费用": "5元", "推荐指数": 0.92},
        {"方式": "公交+地铁", "时间": "45分钟", "费用": "4元", "推荐指数": 0.78},
        {"方式": "打车", "时间": "40分钟", "费用": "55元", "推荐指数": 0.65},
    ]
    
    return {
        "tool": "CDHGNN推荐模型",
        "source": "第四章：对比去偏异构图神经网络",
        "result": {
            "用户画像": f"用户 {user_id}，通勤族，偏好快速到达",
            "推荐方案": recommendations,
            "去偏置信度": 0.87
        }
    }

def tool_route_planning(origin: str, destination: str, mode: str) -> dict:
    """
    🔧 工具5：高德地图路径规划（MCP 外部工具）
    模拟 MCP 调用高德地图 API
    """
    routes = {
        "地铁": {"路线": "9号线 → 10号线 → 知春路站", "距离": "12.5km", "换乘": "1次"},
        "公交": {"路线": "387路 → 地铁4号线", "距离": "14.2km", "换乘": "1次"},
        "驾车": {"路线": "莲花池东路 → 西二环 → 北三环", "距离": "15.8km", "收费": "无"},
    }
    
    return {
        "tool": "高德地图MCP",
        "source": "外部能力：MCP协议调用",
        "result": routes.get(mode, routes["地铁"])
    }

# ============================================================
# 工具注册表（告诉 LLM 有哪些工具可用）
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "traffic_prediction",
            "description": "调用第一章的时空预测模型，预测指定路段的交通状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "起点位置"},
                    "destination": {"type": "string", "description": "终点位置"},
                    "time": {"type": "string", "description": "预测时间"}
                },
                "required": ["origin", "destination", "time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "anomaly_detection",
            "description": "调用第二章的异常感知模型，检测指定区域的交通异常事件",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "检测区域"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "causal_analysis",
            "description": "调用第三章的因果分析模型，分析异常事件的传播影响",
            "parameters": {
                "type": "object",
                "properties": {
                    "affected_area": {"type": "string", "description": "受影响区域"}
                },
                "required": ["affected_area"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "travel_recommendation",
            "description": "调用第四章的CDHGNN模型，为用户推荐最优出行方式",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID"},
                    "origin": {"type": "string", "description": "起点"},
                    "destination": {"type": "string", "description": "终点"}
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_planning",
            "description": "通过MCP协议调用高德地图，获取详细路线规划",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "起点"},
                    "destination": {"type": "string", "description": "终点"},
                    "mode": {"type": "string", "description": "交通方式：地铁/公交/驾车"}
                },
                "required": ["origin", "destination", "mode"]
            }
        }
    }
]

# 工具执行函数映射
TOOL_FUNCTIONS = {
    "traffic_prediction": lambda args: tool_traffic_prediction(args.get("origin", "未知起点"), args.get("destination", "未知终点"), args.get("time", "当前")),
    "anomaly_detection": lambda args: tool_anomaly_detection(args.get("location", "未知区域")),
    "causal_analysis": lambda args: tool_causal_analysis(args.get("affected_area", "未知区域")),
    "travel_recommendation": lambda args: tool_travel_recommendation(args.get("user_id", "U001"), args.get("origin", "未知起点"), args.get("destination", "未知终点")),
    "route_planning": lambda args: tool_route_planning(args.get("origin", "未知起点"), args.get("destination", "未知终点"), args.get("mode", "地铁")),
}

# ============================================================
# 智能体核心逻辑
# ============================================================

SYSTEM_PROMPT = """你是一个智能交通诱导助手，基于博士论文《面向异常态势的超大规模路网交通预测与诱导》中的研究成果构建。

你可以调用以下工具来帮助用户：
1. **时空预测模型**（第一章）：预测路网交通状态
2. **异常感知模型**（第二章）：检测交通异常事件
3. **因果分析模型**（第三章）：分析异常传播影响
4. **CDHGNN推荐模型**（第四章）：推荐最优出行方式
5. **高德地图MCP**：获取详细路线规划

工作流程：
1. 理解用户的出行需求
2. 根据需求调用合适的工具（可以调用多个）
3. 综合工具返回的结果
4. 生成友好、可操作的出行建议报告


请用专业但友好的语气回复，适当使用 emoji 让回复更生动。

⚠️ 重要提示：
如果需要调用多个工具，请务必生成多个独立的工具调用（Tool Calls），绝不要在一个工具调用的参数中拼接多个 JSON 对象！
每个工具调用只需包含该工具所需的参数。"""


def run_agent(user_message: str, history: list) -> str:
    """
    运行智能体：理解用户意图 → 调用工具 → 生成报告
    """
    if not client:
        return "⚠️ 请先设置 OPENAI_API_KEY 环境变量！\n\n设置方法：\n```powershell\n$env:OPENAI_API_KEY = 'sk-xxxxxxxx'\n```"
    
    # 构建消息历史
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": user_message})
    
    # 第一次调用：让 LLM 决定调用哪些工具
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        assistant_message = response.choices[0].message
    except Exception as e:
        error_msg = f"⚠️ API 调用失败：{str(e)}"
        print(f"\n❌ [ERROR] {error_msg}\n")
        return error_msg
    
    # ====== DEBUG: 打印LLM原始输出 ======
    print("\n" + "="*60)
    print("🔍 [DEBUG] LLM 原始输出:")
    print("="*60)
    print(f"Content: {assistant_message.content}")
    if assistant_message.tool_calls:
        print(f"\nTool Calls ({len(assistant_message.tool_calls)}个):")
        for i, tc in enumerate(assistant_message.tool_calls):
            print(f"  [{i+1}] {tc.function.name}")
            print(f"      Arguments: {tc.function.arguments}")
    else:
        print("\nTool Calls: 无")
    print("="*60 + "\n")
    
    # 如果 LLM 决定调用工具
    tool_results = []
    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            args_str = tool_call.function.arguments
            
            # 检测是否有多个JSON对象被拼接（LLM格式错误）
            # 例如: {"a":1}{"b":2} 应该被拆分
            # import re
            # json_objects = re.findall(r'\{[^{}]*\}', args_str)
            
            # 使用更健壮的括号计数法提取 JSON
            json_objects = []
            stack = 0
            start_index = -1
            for i, char in enumerate(args_str):
                if char == '{':
                    if stack == 0:
                        start_index = i
                    stack += 1
                elif char == '}':
                    stack -= 1
                    if stack == 0 and start_index != -1:
                        json_objects.append(args_str[start_index:i+1])
                        start_index = -1
            
            if len(json_objects) > 1:
                print(f"⚠️ [DEBUG] 检测到多个JSON对象被拼接，尝试匹配正确的参数...")
                print(f"   找到 {len(json_objects)} 个JSON对象: {json_objects}")
            
            try:
                func_args = json.loads(args_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ [DEBUG] JSON解析失败: {e}")
                print(f"   原始字符串: {args_str}")
                
                # 根据工具类型选择合适的JSON对象
                func_args = None
                for json_str in json_objects:
                    try:
                        candidate = json.loads(json_str)
                        # 检查这个候选对象是否包含当前工具所需的参数
                        if func_name == "travel_recommendation" and ("origin" in candidate or "destination" in candidate):
                            func_args = candidate
                            print(f"   ✓ 为 {func_name} 匹配到: {func_args}")
                            break
                        elif func_name == "anomaly_detection" and "location" in candidate:
                            func_args = candidate
                            print(f"   ✓ 为 {func_name} 匹配到: {func_args}")
                            break
                        elif func_name == "traffic_prediction" and "origin" in candidate and "destination" in candidate:
                            func_args = candidate
                            print(f"   ✓ 为 {func_name} 匹配到: {func_args}")
                            break
                        elif func_name == "causal_analysis" and "affected_area" in candidate:
                            func_args = candidate
                            print(f"   ✓ 为 {func_name} 匹配到: {func_args}")
                            break
                        elif func_name == "route_planning" and "mode" in candidate:
                            func_args = candidate
                            print(f"   ✓ 为 {func_name} 匹配到: {func_args}")
                            break
                    except:
                        continue
                
                if func_args is None and json_objects:
                    # 如果没有匹配到，使用第一个
                    try:
                        func_args = json.loads(json_objects[0])
                        print(f"   ⚠️ 未能精确匹配，使用第一个JSON: {func_args}")
                    except:
                        print(f"   ❌ 无法解析任何JSON对象")
                        continue
                elif func_args is None:
                    print(f"   ❌ 无可用的JSON对象")
                    continue
            
            # 执行工具
            if func_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[func_name](func_args)
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        
        # 将工具结果发回 LLM，生成最终报告
        messages.append(assistant_message)
        messages.extend(tool_results)
        
        try:
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )
            
            # DEBUG: 打印最终响应详情
            print("-"*60)
            print("🔍 [DEBUG] 最终响应:")
            print(f"Content: {final_response.choices[0].message.content}")
            print(f"Finish Reason: {final_response.choices[0].finish_reason}")
            print("-" * 60)
            
            result = final_response.choices[0].message.content
            if not result:
                return "⚠️ LLM 返回了空内容。这可能是因为模型尝试继续调用工具，但第二轮并未提供工具定义。请重试。"
            return result
        except Exception as e:
            error_msg = f"⚠️ 生成最终报告时 API 调用失败：{str(e)}"
            print(f"\n❌ [ERROR] {error_msg}\n")
            return error_msg
    
    # 如果不需要工具，直接返回
    result = assistant_message.content
    return result if result else "🤔 我不太确定如何回答这个问题，请换个方式提问试试。"


# ============================================================
# Gradio 对话界面
# ============================================================

def create_ui():
    """创建 Gradio 对话界面"""
    
    with gr.Blocks(
        title="🚗 智能交通诱导助手",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 900px !important; }
        .message { font-size: 15px !important; }
        footer { display: none !important; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🚗 智能交通诱导助手
        
        > 基于博士论文《面向异常态势的超大规模路网交通预测与诱导》研究成果
        
        **可用能力：**
        - 📊 时空预测模型（第一章）
        - ⚠️ 异常感知模型（第二章）
        - 🔗 因果分析模型（第三章）
        - 🎯 CDHGNN推荐模型（第四章）
        - 🗺️ 高德地图MCP
        
        ---
        """)
        
        chatbot = gr.Chatbot(
            label="对话记录",
            height=450,
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=agent"),
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="输入您的出行需求",
                placeholder="例如：我明天早上8点要从北京西站去中关村，听说三环有施工，帮我规划一下",
                scale=4,
                show_label=False,
            )
            submit = gr.Button("发送 🚀", variant="primary", scale=1)
        
        with gr.Row():
            gr.Examples(
                examples=[
                    "我想从北京西站去中关村，现在是早高峰，推荐怎么走？",
                    "帮我查一下东三环目前有没有交通异常？",
                    "如果西二环发生事故，会影响到哪些路段？",
                    "预测明天下午5点北京南站到望京的交通状况",
                ],
                inputs=msg,
                label="💡 示例问题"
            )
        
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            
            # 将新格式转换为旧格式供 run_agent 使用
            old_format_history = []
            for item in chat_history:
                if isinstance(item, dict):
                    # 新格式: {"role": "user/assistant", "content": "..."}
                    if item.get("role") == "user":
                        old_format_history.append((item.get("content", ""), ""))
                    elif item.get("role") == "assistant" and old_format_history:
                        last = old_format_history[-1]
                        old_format_history[-1] = (last[0], item.get("content", ""))
                elif isinstance(item, tuple):
                    old_format_history.append(item)
            
            # 运行智能体
            bot_response = run_agent(message, old_format_history)
            
            # 使用新格式添加消息
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_response})
            return "", chat_history
        
        # 绑定事件
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        
        gr.Markdown("""
        ---
        <center>
        
        **技术架构：** LangGraph 认知编排 + OpenAI GPT + MCP 协议 + FastAPI
        
        *本系统为博士论文第四章技术演示*
        
        </center>
        """)
    
    return demo


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🚗 智能交通诱导助手")
    print("="*60)
    
    if not OPENAI_API_KEY:
        print("\n⚠️  警告：未检测到 OPENAI_API_KEY")
        print("   请设置环境变量：")
        print("   $env:OPENAI_API_KEY = 'sk-xxxxxxxx'")
        print("\n   界面仍会启动，但无法进行真实对话。\n")
    else:
        print(f"\n✅ 已检测到 API Key（以 {OPENAI_API_KEY[:8]}... 开头）\n")
    
    print("🌐 正在启动 Gradio 界面...")
    print("   访问地址：http://localhost:7860")
    print("   按 Ctrl+C 停止服务\n")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设为 True 可生成公网链接
        show_error=True
    )
