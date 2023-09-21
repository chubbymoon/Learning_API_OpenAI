import openai
import json
import inspect
import os
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")


class AutoFunctionGenerator:
    def __init__(self, functions_list, max_attempts=2):
        self.functions_list = functions_list
        self.max_attempts = max_attempts

    def generate_function_descriptions(self):
        """生成功能描述

        :return: 每个功能函数的JSON Schema描述
        """
        # 创建空列表，保存每个功能函数的JSON Schema描述
        functions = []

        for function in self.functions_list:
            # 读取指定函数的函数说明
            function_description = inspect.getdoc(function)

            # 读取函数的函数名
            function_name = function.__name__

            # 定义system role的Few-shot提示
            system_Q = "你是一位优秀的数据分析师，现在有一个函数的详细声明如下：%s" % function_description
            system_A = "计算年龄总和的函数，从给定的JSON格式字符串（按'split'方向排列）中解析出DataFrame，计算所有人的年龄总和，并以JSON格式返回结果。\
                        \n:param input_json: 必要参数，要求字符串类型，表示含有个体年龄数据的JSON格式字符串 \
                        \n:return: 所有人的年龄总和，以 JSON 格式返回。"

            # 定义user role的Few-shot提示
            user_Q = "请根据这个函数声明，为我生成一个JSON Schema对象描述。这个描述应该清晰地标明函数的输入和输出规范。具体要求如下：\
                      1. 提取函数名称：%s，并将其用作JSON Schema中的'name'字段  \
                      2. 在JSON Schema对象中，设置函数的参数类型为'object'.\
                      3. 'properties'字段如果有参数，必须表示出字段的描述. \
                      4. 从函数声明中解析出函数的描述，并在JSON Schema中以中文字符形式表示在'description'字段.\
                      5. 识别函数声明中哪些参数是必需的，然后在JSON Schema的'required'字段中列出这些参数. \
                      6. 输出的应仅为符合上述要求的JSON Schema对象内容,不需要任何上下文修饰语句. " % function_name

            user_A = "{'name': 'calculate_total_age_function', \
                       'description': '计算年龄总和的函数，从给定的JSON格式字符串（按'split'方向排列）中解析出DataFrame，计算所有人的年龄总和，并以JSON格式返回结果。 \
                       'parameters': {'type': 'object', \
                                      'properties': {'input_json': {'description': '表示含有个体年龄数据的JSON格式字符串', 'type': 'string'}}, \
                                      'required': ['input_json']}, \
                       'returns': {'type': 'string', \
                                   'description':'所有人的年龄总和，以 JSON 格式返回。'}\
                       }"

            # 定义输入
            system_message = "你是一位优秀的数据分析师，现在有一个函数的详细声明如下：%s" % function_description
            user_message = "请根据这个函数声明，为我生成一个JSON Schema对象描述。这个描述应该清晰地标明函数的输入和输出规范。具体要求如下：\
                            1. 提取函数名称：%s，并将其用作JSON Schema中的'name'字段  \
                            2. 在JSON Schema对象中，设置函数的参数类型为'object'.\
                            3. 'properties'字段如果有参数，必须表示出字段的描述. \
                            4. 从函数声明中解析出函数的描述，并在JSON Schema中以中文字符形式表示在'description'字段.\
                            5. 识别函数声明中哪些参数是必需的，然后在JSON Schema的'required'字段中列出这些参数. \
                            6. 输出的应仅为符合上述要求的JSON Schema对象内容,不需要任何上下文修饰语句. " % function_name

            messages = [
                {"role": "system", "content": "Q:" + system_Q + user_Q + "A:" + system_A + user_A},

                {"role": "user", "content": 'Q:' + system_message + user_message}
            ]

            response = self._call_openai_api(messages)
            functions.append(json.loads(response.choices[0].message['content']))
        return functions

    def _call_openai_api(self, messages):
        # 请根据您的实际情况修改此处的 API 调用
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=messages,
        )
        return response

    def auto_generate(self):
        # 记录尝试次数
        attempts = 0
        while attempts < self.max_attempts:
            print(" Please wait...")
            try:
                functions = self.generate_function_descriptions()
                return functions
            except Exception as e:
                attempts += 1
                print(f"Error occurred: {e}")
                if attempts >= self.max_attempts:
                    print("Reached maximum number of attempts. Terminating.")
                    raise
                else:
                    print(" Retrying...")


class ChatConversation:
    """
    ChatConversation 类用于与 OpenAI GPT-3 模型进行聊天对话，并可选地调用外部功能函数。

    属性:
    - model (str): 使用的 OpenAI GPT模型名称。
    - messages (list): 存储与 GPT 模型之间的消息。
    - function_repository (dict): 存储可选的外部功能函数。

    方法:
    - __init__ : 初始化 ChatConversation 类。
    - add_functions : 添加外部功能函数到功能仓库。
    - _call_chat_model : 调用 OpenAI GPT 模型进行聊天。
    - run : 运行聊天会话并获取最终的响应。
    """

    def __init__(self, model="gpt-3.5-turbo-16k-0613"):
        """
        初始化ChatConversation类。
        """
        self.model = model
        self.messages = []
        self.function_repository = {}
        # 使用模型 "gpt-3.5-turbo-0613" 会出现报错: Error calling chat model: Rate limit reached for default-gpt-3.5-turbo in organization org-8k1NKQgoSqBcuRs0LqkUOrOp on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.

    def add_functions(self, functions_list):
        """
        添加功能函数到功能仓库。

        参数:
        functions_list (list): 包含功能函数的列表。
        """
        self.function_repository = {func.__name__: func for func in functions_list}

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
            "messages": self.messages,
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        try:
            return openai.ChatCompletion.create(**params)
        except Exception as e:
            print(f"Error calling chat model: {e}")
            return None

    def run(self, functions_list=None):
        """
        运行聊天会话，可能包括外部功能函数调用。

        参数:
        functions_list (list): 包含功能函数的列表。如果为 None，则只进行常规对话。

        返回:
        str: 最终的聊天模型响应。
        """
        try:
            # 如果不传入外部函数仓库，就进行常规的对话
            if functions_list is None:
                response = self._call_chat_model()
                final_response = response["choices"][0]["message"]["content"]
                return final_response

            else:

                # 添加功能函数到功能仓库
                self.add_functions(functions_list)

                # 如果存在外部的功能函数，生成每个功能函数对应的JSON Schema对象描述
                functions = AutoFunctionGenerator(functions_list).auto_generate()

                # 第一次调用大模型，获取到 first response
                response = self._call_chat_model(functions, include_functions=True)
                response_message = response["choices"][0]["message"]

                # 检查在 first response 中是否存在function_call，如果存在，说明需要调用到外部函数仓库
                if "function_call" in response_message:

                    # 获取函数名
                    function_name = response_message["function_call"]["name"]

                    # 获取函数对象
                    function_call_exist = self.function_repository.get(function_name)

                    if not function_call_exist:
                        print(f"Function {function_name} not found in functions repository.")
                        return None

                    # 获取函数关键参数信息
                    function_args = json.loads(response_message["function_call"]["arguments"])

                    # 获取函数逻辑处理后的结果
                    function_response = function_call_exist(**function_args)

                    # messages = 原始输入 + first reponse + function_response

                    # messages中拼接 first response 消息
                    self.messages.append(response_message)

                    # messages中拼接函数输出结果
                    self.messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response,
                        }
                    )

                    # 第二次调用模型
                    second_response = self._call_chat_model()

                    # 获取最终的计算结果
                    final_response = second_response["choices"][0]["message"]["content"]

                else:
                    final_response = response_message["content"]

                return final_response

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


# ChatConversation 测试_1: 不带入外部函数仓库
if __name__ == '__main__' and 0:
    # 提供数据
    df_complex = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000.0, 100000.5, 150000.75],
        'IsMarried': [True, False, True]
    })
    df_complex_json = df_complex.to_json(orient='split')

    # 创建一个 ChatConversation 实例
    conv = ChatConversation()

    conv.messages = [
        {"role": "system", "content": "你是一位优秀的数据分析师，现在有这样一个数据集 input_json：% s，数据集以 JSON 形式呈现" % df_complex_json},
        {"role": "user", "content": "请在数据集 input_json 上执行计算所有人年龄总和函数"}
    ]
    # 运行对话
    result = conv.run()
    print(result)
    """ result 输出示例:
    您可以使用 Python 解析这个 JSON 数据集，并计算所有人的年龄总和。下面是一个示例代码：
    
    ```python
    import json
    
    input_json = '{"columns":["Name","Age","Salary","IsMarried"],"index":[0,1,2],"data":[["Alice",25,50000.0,true],["Bob",30,100000.5,false],["Charlie",35,150000.75,true]]}'
    
    # 解析 JSON 数据
    data = json.loads(input_json)
    
    # 获取年龄列的索引
    age_index = data['columns'].index('Age')
    
    # 计算年龄总和
    age_sum = sum([row[age_index] for row in data['data']])
    
    print("所有人的年龄总和为:", age_sum)
    ```
    
    输出结果为：
    
    ```
    所有人的年龄总和为: 90
    ```
    
    进程已结束，退出代码为 0
    
    """

# ChatConversation 测试_2: 不带入外部函数仓库
if __name__ == '__main__':
    #  # 提供数据
    df_complex = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000.0, 100000.5, 150000.75],
        'IsMarried': [True, False, True]
    })
    df_complex_json = df_complex.to_json(orient='split')


    # 测试函数1
    def calculate_total_age_function(input_json):
        """
        从给定的JSON格式字符串（按'split'方向排列）中解析出DataFrame，计算所有人的年龄总和，并以JSON格式返回结果。

        参数:
        input_json (str): 包含个体数据的JSON格式字符串。

        返回:
        str: 所有人的年龄总和，以JSON格式返回。
        """

        # 将JSON字符串转换为DataFrame
        df = pd.read_json(input_json, orient='split')

        # 计算所有人的年龄总和
        total_age = df['Age'].sum()

        # 将结果转换为字符串形式，然后使用json.dumps()转换为JSON格式
        return json.dumps({"total_age": str(total_age)})


    # 测试函数2
    def calculate_married_count(input_json):
        """
        从给定的JSON格式字符串中解析出DataFrame，计算结婚人数，并以JSON格式返回结果。

        参数:
        input_json (str): 包含个体数据（其中包括婚姻状态）的JSON格式字符串。

        返回:
        str: 结婚人数，以JSON格式返回。
        """

        # 将JSON字符串转换为DataFrame
        df = pd.read_json(input_json, orient='split')

        # 计算结婚人数
        married_count = df[df['IsMarried'] == True].shape[0]

        # 将结果转换为字符串形式，然后使用json.dumps()转换为JSON格式
        return json.dumps({"married_count": str(married_count)})


    # 定义函数列表
    functions_list = [calculate_total_age_function, calculate_married_count]

    # 创建一个 ChatConversation 实例
    conv = ChatConversation()
    # 注意: 需使用 16k 模型 (gpt-3.5-turbo-16k-0613), 使用模型 "gpt-3.5-turbo-0613" 会出现如下报错:
    """
     Error calling chat model: 
    Rate limit reached for default-gpt-3.5-turbo in organization org-8k1NXXgoSqBxxRs0LqkUXxOx on requests per min. 
    Limit: 3 / min. Please try again in 20s. 
    Contact us through our help center at help.openai.com if you continue to have issues. 
    Please add a payment method to your account to increase your rate limit. 
    Visit https://platform.openai.com/account/billing to add a payment method.
    """

    conv.messages = [
        {"role": "system", "content": "你是一位优秀的数据分析师，现在有这样一个数据集 input_json：% s，数据集以 JSON 形式呈现" % df_complex_json},
        {"role": "user", "content": "请在数据集 input_json 上执行计算所有人年龄总和函数"}
    ]

    # 运行对话
    result = conv.run(functions_list=functions_list)
    print(result)
    """ result 输出示例
    在数据集 input_json 中，所有人年龄的总和为 90。
    """

# TODO 1.增加调试模块 (了解过程), 2.增加等待提示(避免空白等待), 3.增加 "流式输出" 或 整体输出 的选择
