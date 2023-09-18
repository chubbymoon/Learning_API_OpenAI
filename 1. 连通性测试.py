import openai
import os

# 获取系统变量(windows)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 创建一个 GPT-3 请求
completion = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo-0613",
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, Nice to meet you"}]
)



# print(completion.choices[0].message)
print(completion)
"""
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
