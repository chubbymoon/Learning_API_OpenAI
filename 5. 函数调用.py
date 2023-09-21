import json
import sys
import pandas as pd
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

"""1. 数据"""
# 示例 DataFrame
df_complex = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000.0, 100000.5, 150000.75],
    'IsMarried': [True, False, True]
})
# 将 DataFrame 转换为 JSON 格式（按'split' 方向）
df_complex_json = df_complex.to_json(orient='split')
print(f"df_complex_json: {df_complex_json}")
# df_complex_json: {"columns":["Name","Age","Salary","IsMarried"],"index":[0,1,2],"data":[["Alice",25,50000.0,true],["Bob",30,100000.5,false],["Charlie",35,150000.75,true]]}


"""2. 设定需求"""
# 让大模型计算这个数据集中所有人的年龄总和。
print(f"user: 请在数据集 input_json: df_complex_json 上执行计算所有人年龄总和函数")


"""3. 定义功能函数"""
# 编写计算年龄总和的函数
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


"""4. 检验功能函数"""
# 使用函数计算年龄总和，并以 JSON 格式输出
result = calculate_total_age_from_split_json(df_complex_json)
# print("The JSON output is:", result)
# The JSON output is: {"total_age": "90"}


"""5. 定义函数库"""
# 将功能函数存储至外部函数仓库
function_repository = {
    "calculate_total_age_from_split_json": calculate_total_age_from_split_json,
}


"""6. 创建功能函数的 JSON Schema"""
# 5.与 6. 的顺序不能颠倒, 函数的 JSON Schema 与定义的功能函数重名
calculate_total_age_from_split_json = {"name": "calculate_total_age_from_split_json",
                                       "description": "计算年龄总和的函数，从给定的 JSON 格式字符串（按'split' 方向排列）中解析出 DataFrame，计算所有人的年龄总和，并以 JSON 格式返回结果。",
                                       "parameters": {"type": "object",
                                                      "properties": {"input_json": {"type": "string",
                                                                                    "description": "执行计算年龄总和的数据集"},
                                                                     },
                                                      "required": ["input_json"],
                                                      },
                                       }


"""7. 创建函数列表"""
# 添加到 functions 列表中，在对话过程中作为函数库传递给 function 参数
functions = [calculate_total_age_from_split_json]


"""8. 创建上下文列表(messages)"""
messages = [
    {"role": "system", "content": f"你是一位优秀的数据分析师，现在有这样一个数据集 input_json：{df_complex_json}，数据集以 JSON 形式呈现"},
    {"role": "user", "content": "请在数据集 input_json 上执行计算所有人年龄总和函数"},
    # 不同的提问测试
    # {"role": "user", "content": "请在给定的数据上计算所有人年龄总和"},    # 依然会调用函数
    # {"role": "user", "content": "请分析给定的数据"},    # 不会调用函数
]


"""9. 将用户提问传入 GPT 模型，让其自动选择函数和相关参数"""
try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
except Exception as e:
    print(e)
    print("程序已结束...")
    sys.exit()
    pass

# print(f"response: {response}")
"""  函数调用 response 输出示例:
response: {
  "id": "chatcmpl-80ryKEMonPeyUXid45ROEZnIDOkWJ",
  "object": "chat.completion",
  "created": 1695218160,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "calculate_total_age_from_split_json",
          "arguments": "{\n  \"input_json\": \"{\\\"columns\\\":[\\\"Name\\\",\\\"Age\\\",\\\"Salary\\\",\\\"IsMarried\\\"],\\\"index\\\":[0,1,2],\\\"data\\\":[[\\\"Alice\\\",25,50000.0,true],[\\\"Bob\\\",30,100000.5,false],[\\\"Charlie\\\",35,150000.75,true]]}\"\n}"
        }
      },
      "finish_reason": "function_call"
    }
  ],
  "usage": {
    "prompt_tokens": 225,
    "completion_tokens": 82,
    "total_tokens": 307
  }
}
"""

"""10. 保存 GPT 返回的关键信息(需要调用的函数和相关参数)"""
# TODO 不能保证 GPT 一定会调用函数, 需要异常处理
# 保存交互过程中的函数名称
function_name = response["choices"][0]["message"]["function_call"]["name"]
# 加载交互过程中的参数
function_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])

# print(f"function_name: {function_name}")
# print(f"function_args: {function_args}")
# function_name: calculate_total_age_from_split_json
# function_args: {'input_json': '{"columns":["Name","Age","Salary","IsMarried"],"index":[0,1,2],"data":[["Alice",25,50000.0,true],["Bob",30,100000.5,false],["Charlie",35,150000.75,true]]}'}


"""11. 保存函数对象"""
# 保存具体的函数对象
local_fuction_call = function_repository[function_name]

# print(f"local_fuction_call: {local_fuction_call}")
# local_fuction_call: <function calculate_total_age_from_split_json at 0x000001590BE6D310>


"""12. 调用本地函数处理 GPT 返回的关键信息"""
final_response = local_fuction_call(**function_args)

# print(f"final_response: {final_response}")
# final_response: {"total_age": "90"}


"""13. 将本地函数运行的结果追加到上下文 (messages) 中"""
# 追加第一次模型返回结果消息
messages.append(response["choices"][0]["message"])
# 追加 function 计算结果，注意：function message 必须要输入关键词 name
messages.append({"role": "function", "name": function_name, "content": final_response, })


"""14. 将包含本地函数运行的结果的上下文 (messages) 传入 GPT, 获取结果"""
try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
    )
except Exception as e:
    print(e)
    print("程序已结束...")
    sys.exit()
    pass


"""15. 获取结果"""
print("GPT: ", response.choices[0].message['content'])
