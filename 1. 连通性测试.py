import openai
import os

# 获取系统变量(windows)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 创建一个 GPT-3 请求
response = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo-0613",
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, Nice to meet you"}]
)

# 输出完整的响应 , 主体内容 (content) 以 Unicode 编码
# print(response)
""" response 输出示例
{
  "id": "chatcmpl-7yy2jCJGBDBaiZkinTzuAiTJIth5U",
  "object": "chat.completion",
  "created": 1694764841,
  "model": "gpt-3.5-turbo-0613",
  "choices": [3
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! Nice to meet you too. How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 15,
    "total_tokens": 28
  }
}

"""


# 响应的数据类型
# print(type(response))
# <class 'openai.openai_object.OpenAIObject'>

# 输出响应的主体内容 (content), 主体内容为可读文本(汉语)
print(response.choices[0].message['content'])
# 输出本次对话消耗的总的 Token 数 (提示词与响应 Token 之和)
print(response.usage['total_tokens'])
