import random
from langchain_core.tools import tool

@tool
def traffic_prediction(origin: str, destination: str, time: str = "当前"):
    """
    第一章：时空预测模型。预测指定路段在特定时间的交通拥堵指数和速度。
    """
    return {
        "tool": "时空预测模型",
        "source": "第一章：基于Transformer的路网时空预训练",
        "result": {
            "拥堵指数": round(random.uniform(0.3, 0.9), 2),
            "预测速度": f"{random.randint(20, 60)} km/h",
            "置信度": 0.89,
            "备注": f"预测 {time} 从 {origin} 到 {destination} 的交通状态"
        }
    }

@tool
def anomaly_detection(location: str):
    """
    第二章：异常感知模型。检测指定区域是否存在交通事故或道路施工等异常事件。
    """
    anomalies = [
        {"类型": "交通事故", "位置": location, "影响时长": "约2小时", "严重程度": "中等"},
        {"类型": "道路施工", "位置": location, "影响时长": "持续至本周五", "严重程度": "轻微"},
        {"类型": "无异常", "位置": location, "影响时长": "-", "严重程度": "-"}
    ]
    return {
        "tool": "异常感知模型",
        "source": "第二章：融合LLM的多模态异常感知",
        "result": random.choice(anomalies)
    }

@tool
def causal_analysis(affected_area: str):
    """
    第三章：因果分析模型。分析异常事件对周边路网的传播影响和因果强度。
    """
    return {
        "tool": "因果分析模型",
        "source": "第三章：基于几何深度学习的动态因果发现",
        "result": {
            "影响传播路径": f"{affected_area} → 周边辅路 → 邻近主干道",
            "预计波及时间": "30-45分钟",
            "因果强度": round(random.uniform(0.6, 0.9), 2),
            "建议绕行": "选择非波及区域的快速路"
        }
    }

@tool
def travel_recommendation(origin: str, destination: str, user_id: str = "U001"):
    """
    第四章：CDHGNN推荐模型。基于异构图神经网络为用户推荐最优出行方案。
    """
    recommendations = [
        {"方式": "地铁", "时间": "35分钟", "费用": "5元", "推荐指数": 0.92},
        {"方式": "驾车", "时间": "45分钟", "费用": "50元", "推荐指数": 0.75},
    ]
    return {
        "tool": "CDHGNN推荐模型",
        "source": "第四章：对比去偏异构图神经网络",
        "result": {
            "用户画像": f"用户 {user_id}，偏好效率优先",
            "推荐方案": recommendations,
            "去偏置信度": 0.87
        }
    }
