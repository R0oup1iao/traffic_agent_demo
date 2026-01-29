from openai import OpenAI
import os

# 初始化客户端
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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", # 建议放入环境变量中
)

try:
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "http://localhost:8000", # 可选：你的网站URL
            "X-Title": "LocalDev", # 可选：你的应用名称
        },
        # 关键点：因为没充钱，必须使用免费模型
        # 免费模型通常以 :free 结尾，或者在官网标注为 Free
        model="xiaomi/mimo-v2-flash:free", 
        messages=[
            {
                "role": "user",
                "content": "你好，请介绍一下你自己。"
            }
        ]
    )
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"发生错误: {e}")