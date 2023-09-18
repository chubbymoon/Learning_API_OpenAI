import inspect
import json
import os
import openai
import pandas as pd


openai.api_key = os.getenv("OPENAI_API_KEY")


# 测试的功能函数
def calculate_total_age_from_split_json(input_json):
    """
    从给定的 JSON 格式字符串（按'split' 方向排列）中解析出 DataFrame，计算所有人的年龄总和，并以 JSON 格式返回结果。

    参数:
    input_json (str): 包含个体数据的 JSON 格式字符串。

    返回:
    str: 所有人的年龄总和，以 JSON 格式返回。
    """

    # 将 JSON 字符串转换为 DataFrame
    df = pd.read_json(input_json, orient='split')

    # 计算所有人的年龄总和
    total_age = df['Age'].sum()

    # 将结果转换为字符串形式，然后使用 json.dumps () 转换为 JSON 格式
    return json.dumps({"total_age": str(total_age)})


# 使用 inspect 模块提取文档字符串
function_declaration = inspect.getdoc(calculate_total_age_from_split_json)
# print(function_declaration)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "system", "content": "你是一位优秀的数据分析师，现在有一个函数的详细声明如下：%s" % function_declaration},
        {"role": "user", "content": "请根据这个函数声明，为我生成一个JSON Schema对象描述。这个描述应该清晰地标明函数的输入和输出规范。具体要求如下：\
                                1. 在JSON Schema对象中，设置函数的参数类型为'object'.\
                                2. 'properties'字段如果有参数，必须表示出字段的描述. \
                                3. 从函数声明中解析出函数的描述，并在JSON Schema中以中文字符形式表示在'description'字段.\
                                4. 识别函数声明中哪些参数是必需的，然后在JSON Schema的'required'字段中列出这些参数. \
                                5. 输出的应仅为符合上述要求的JSON Schema对象内容,不需要任何上下文修饰语句. "}
    ]
)

print(response.choices[0].message['content'])
""" response.choices[0].message['content']: 
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "input_json": {
      "type": "string",
      "description": "包含个体数据的 JSON 格式字符串"
    }
  },
  "required": ["input_json"],
  "description": "从给定的 JSON 格式字符串（按'split' 方向排列）中解析出 DataFrame，计算所有人的年龄总和，并以 JSON 格式返回结果。",
  "returns": {
    "type": "string",
    "description": "所有人的年龄总和，以 JSON 格式返回"
  }
}
"""

# 通过对比手动编写的结果，两者是高度一致的，这就验证了模型能够根据函数的参数说明正确识别计算函数的参数