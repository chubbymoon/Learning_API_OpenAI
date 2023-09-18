import json
import sys
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

"""1. 定义角色库"""
roles = {"医生": """请以一位专业医生的身份回复""",

         "代码程序员(python)": """请以一位专业程序员的身份回复""",

         "小红书文案助理": """你是小红书文案 AI 私人助理，可以帮助用户快速生成吸引人的小红书文案。""",

         "陪聊": """你是 Al 智能聊天机器人，你很能聊，不管问什么，你都可以给予对方有趣而准确的回复""",

         "其他": "",
         }


"""2. 设定需求"""
# 模拟用户的提示语
user_prompt = "我感冒已经超过7天了，你能帮我分析一下原因吗？"
# user_prompt = "使用 Python 匿名函数将字典里的所有键输出"
print(f"user: {user_prompt}")


"""3. 定义功能函数"""
# 定义功能函数
def get_ability(role):
    """通过不同角色的指定输出相关的提示词, 用于增强 GPT 的回复能力

    通过指定不同角色或能力来输出相关的指导词, 用于应对不同的场景。

    :param role: 指定某种角色或能力
    :return: 输出相关角色的指导词
    """
    if role in roles:
        prompt = roles[role]
    else:
        prompt = ""

    return prompt


"""4. 检验功能函数"""
function_args1 = {"role": "医生"}
result = get_ability(**function_args1)
# print("get_ability:", result)
# get_ability: 请以一位医生的身份回复


"""5. 定义函数库"""
# 将功能函数存储至外部函数仓库
function_repository = {
    "get_ability": get_ability,
}


"""6. 创建功能函数的 JSON Schema"""
# 5.与 6. 的顺序不能颠倒, 函数的 JSON Schema 与定义的功能函数重名
get_ability = {"name": "get_ability",
               "description": f"通过指定不同角色或能力来输出相关的指导词, 用于应对不同的场景。支持的角色或能力有：{list(map(lambda x: x, roles.keys()))}。",
               "parameters": {"type": "object",
                              "properties": {"role": {"type": "string",
                                                      "description": "用于指定某种角色或能力"},
                                             },
                              "required": ["role"],
                              },
               }

"""7. 创建函数列表"""
# 添加到 functions 列表中，在对话过程中作为函数库传递给 function 参数
functions = [get_ability]


"""8. 创建上下文列表(messages)"""
messages = [
    {"role": "system", "content": f"你有着不同的身份和能力{list(map(lambda x: x, roles.keys()))}，请根据他人的需求，选择合适的能力和身份来应对。"},
    {"role": "user", "content": f"{user_prompt}"},

    # 不同的提问测试
    # {"role": "user", "content": "你好!"},    # 不会调用函数
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


"""10. 保存 GPT 返回的关键信息(需要调用的函数和相关参数)"""
# TODO 不能保证 GPT 一定会调用函数, 需要异常处理
# 保存交互过程中的函数名称
function_name = response["choices"][0]["message"]["function_call"]["name"]
# 加载交互过程中的参数
function_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])

# print(f"function_name: {function_name}")
# print(f"function_args: {function_args}")
# function_name: get_ability
# function_args: {"role": "医生"}


"""11. 保存函数对象"""
# 保存具体的函数对象
local_fuction_call = function_repository[function_name]

# print(f"local_fuction_call: {local_fuction_call}")
# local_fuction_call: <function get_ability at 0x0000027B893D53A0>


"""12. 调用本地函数处理 GPT 返回的关键信息"""
final_response = local_fuction_call(**function_args)

# print(f"final_response: {final_response}")
# final_response: 请以一位医生的身份回复


"""13. 将本地函数运行的结果追加到上下文 (messages) 中"""
# 追加第一次模型返回结果消息
# messages.append(response["choices"][0]["message"])
# 追加 function 计算结果，注意：function message 必须要输入关键词 name
# messages.append({"role": "function", "name": function_name, "content": final_response, })
messages.append({"role": "system", "content": final_response, })

print(f"messages: {messages}")
# messages: [{'role': 'system', 'content': '你有着不同的身份和不同的能力，请根据他人的需求，选择合适的能力和身份来应对。'}, {'role': 'user', 'content': '我感冒已经超过7天了，你能帮我分析一下原因吗？'}, {'role': 'system', 'content': '请以一位专业医生的身份回复'}]


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
