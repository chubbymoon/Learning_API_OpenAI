import json
import time
import types
import openai
import os
import tiktoken
from logger import logger

openai.api_key = os.getenv("OPENAI_API_KEY")


class CL_GPT:
    """ CL_GPT 类用于与 OpenAI GPT-3 模型进行聊天对话, 不过是在命令行中使用它，而且它可以调用外部功能函数。

    输入 STOP 即可结束聊天

    # TODO 调用函数未功能测试

    属性:
    - model (str): 使用的 OpenAI GPT模型名称, 默认为 "gpt-3.5-turbo-16k-0613"
    - stream (bool): 控制是否开启"流式输出"
    - function_repository (dict): 存储可选的外部功能函数。
    - response (OpenAIObject): 存储 GPT 模型最近一次响应
    - contexts (list): 存储与 GPT 模型之间的消息 (上下文)。
    - token_count (int): 存储最近一次响应的使用的令牌数
    - accumulate_token_count (int): 存储累计消耗的令牌数

    方法:
    - __init__ : 初始化 Chat 类。
    - _call_chat_model : 向 openai 发起请求, 调用大模型
    - _get_response_data : 捕获响应信息
    - show_message : 向用户展示信息
    - run : 运行聊天会话并获取最终的响应。
    - _join_contexts : 将某条信息加入到上下文中
    - bubble : 配置消息气泡
    """

    def __init__(self, model="gpt-3.5-turbo-16k-0613", stream=False):
        """
        初始化Chat类。
        """
        # 模型
        self.model = model
        # 是否开启"流式输出"
        self.stream = stream
        # 函数库
        self.function_repository = {}
        # 最近一次响应
        self.response = None
        # 上下文
        self.contexts = []
        # 最近一次响应的使用的令牌数
        self.token_count = 0
        # 累计令牌数
        self.accumulate_token_count = 0

    def run(self):
        switch = True
        print("-GPT: 你好!")

        while switch:
            self.bubble('user_a')
            user_content = str(input())
            self.bubble('user_b')

            if user_content == "STOP":
                switch = False
                break

            user_message = {"role": "user", "content": user_content}
            self._join_contexts(user_message)
            self._call_chat_model()
            self.show_message()
            pass

    # 向 openai 发起请求, 调用大模型
    def _call_chat_model(self, functions=None, include_functions=False):
        """
        调用大模型。

        参数:
        functions (dict): 功能函数的描述。
        include_functions (bool): 是否包括功能函数和自动功能调用。

        返回:
        dict: 大模型的响应。
        """
        params = {
            "model": self.model,
            "messages": self.contexts,
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        # 控制"流式输出"
        if self.stream:
            params['stream'] = self.stream

        try:
            logger.debug(f"尝试调用 GPT, 参数: {params}")
            logger.info("【提示】等待 GPT 回复, 请稍等...")
            self.response = openai.ChatCompletion.create(**params)
            return self.response
        except Exception as e:
            print(f"Error calling chat model: {e}")
            return None

    # 捕获响应信息
    def _get_response_data(self, response):
        # 处理 "流式输出"
        if type(response) == types.GeneratorType:
            # 默认流式输出及用于对用户呈现, 不做 function_call等的处理使用

            logger.debug("GPT 开始\"流式输出\"...")

            content = ''
            self.bubble('GPT_a')
            try:
                for i in response:
                    cell_content = i.choices[0].delta.get('content')
                    # "流式输出" 最后一项 "content" 为空
                    if cell_content:
                        print(cell_content, end="")
                        content = content + cell_content

                self.bubble('GPT_b')

                logger.debug("已捕获 GPT 响应(流式)...")

            except Exception as e:
                print(e)

            response_message = {"role": "assistant", "content": content}
            self._join_contexts(response_message)

            # 计算本次对话消耗的 token 数, 上下文 + 本次响应的内容
            self.token_count = len(tiktoken.encoding_for_model(self.model).encode(str(self.contexts)))


        # 处理整体输出
        else:
            response_message = response["choices"][0]["message"]
            # 检查在 first response 中是否存在 function_call
            if "function_call" in response_message:
                # 获取函数名
                function_name = response_message["function_call"]["name"]
                # 获取函数参数 (json化)
                # function_args = json.loads(response_message["function_call"]["arguments"])
                # 获取函数参数
                function_args = response_message["function_call"]["arguments"]

                # 获取函数对象 (确保 GPT 回复的函数名正确)
                callable_function = self.function_repository.get(function_name)
                if not callable_function:
                    print(f"Function {function_name} not found in functions repository.")
                    return None

                # 简化响应
                response_message = {"role": "assistant", "function_name": function_name, "function_args": function_args}
                # 记录本次对话消耗的 token 数
                self.token_count = response.usage['total_tokens']

                logger.debug(f"GPT 提出需要调用的函数及参数: {response_message}")

                # 加入上下文
                self._join_contexts(response_message)

                logger.debug("已捕获 GPT 响应(函数调用)...")

            else:
                # 简化响应
                response_message = {"role": "assistant", "content": response_message["content"]}
                # 记录本次对话消耗的 token 数
                self.token_count = response.usage['total_tokens']

                # 加入上下文
                self._join_contexts(response_message)

                logger.debug("已捕获 GPT 响应(整体式)...")

        # 记录 累计消耗的 token
        self.accumulate_token_count = self.accumulate_token_count + self.token_count

        logger.info(f"本次对话消耗的 token 数: {self.token_count}")
        logger.info(f"累计消耗的 token 数: {self.accumulate_token_count}")
        logger.debug(f"当前上下文: {self.contexts}")

        return response_message

    # 向用户展示信息
    def show_message(self):
        # 捕获响应信息
        response_data = self._get_response_data(self.response)

        # 函数调用
        if 'function_name' in response_data:
            callable_function = self.function_repository.get(response_data['function_name'])
            callable_args = json.loads(response_data['function_args'])

            # 获取函数逻辑处理后的结果
            function_response = callable_function(**callable_args)

            # messages中拼接 first response 消息 (感觉没必要) 在<捕获响应信息>时, 已经以简化的形式加入
            # self.messages.append(self.response["choices"][0]["message"])

            # 第二次发送信息
            second_message = {
                "role": "function",
                "name": response_data['function_name'],
                "content": function_response,
            }
            self._join_contexts(second_message)
            logger.debug(f"【函数调用】第二次发送信息: {second_message}")

            # 第二次调用模型
            self._call_chat_model()
            second_response = self._get_response_data(self.response)

        # 正常输出
        elif 'content' in response_data:
            if not self.stream:
                logger.debug("整体式输出中...")
                self.bubble('GPT_a')
                print(response_data['content'], end="")
                self.bubble('GPT_b')

    def _join_contexts(self, message):
        logger.debug("上下文中加入一条消息!")
        self.contexts.append(message)
        pass

    # 设置消息气泡
    def bubble(self, style=None):
        if style == 'user_a':
            time.sleep(0.1)
            print("===" * 10, end="\n\n")
            print("-user: ", end="")
        elif style == 'user_b':
            print("")
            time.sleep(0.1)

        elif style == 'GPT_a':
            time.sleep(0.1)
            print("===", end="\n\n")
            print("-GPT: ", end="")
        elif style == 'GPT_b':
            print("", end="\n\n")
            time.sleep(0.1)
        elif not style:
            pass
        else:
            raise ValueError("对气泡(bubble)指定是种类值错误")


if __name__ == '__main__':
    chat = CL_GPT(stream=True)
    chat.run()
