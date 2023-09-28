import inspect
import json
import os
import openai
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")


class AutoFunctionGenerator:
    """ 自动生成函数描述

    读取函数列表中函数的函数说明, 通过 GPT 转为 JSON Schema 的形式用于后续 GPT 的函数调用.
    如果指定文件路径, 会将结果同时输出到指定文件中

    """
    def __init__(self, functions_list, max_attempts=2, output_path=None):
        self.functions_list = functions_list
        self.max_attempts = max_attempts
        self.output_path = output_path

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
                function_describe_list = self.generate_function_descriptions()
                # 路径存在时将结果输出到文件
                self.output2file(function_describe_list)
                return function_describe_list
            except Exception as e:
                attempts += 1
                print(f"Error occurred: {e}")
                if attempts >= self.max_attempts:
                    print("Reached maximum number of attempts. Terminating.")
                    raise
                else:
                    print(" Retrying...")

    def output2file(self, function_describe_list):
        # 获取文件路径
        file_path = self.output_path
        if not file_path:
            return

        contents = json.dumps(function_describe_list, ensure_ascii=False)

        # 写文件
        try:
            with open(file_path, mode='w', encoding='utf-8') as f:
                f.write(contents)
                pass
        # 处理路径不存在
        except FileNotFoundError:
            log_file_location = os.path.dirname(file_path)
            # 创建路径
            if not os.path.exists(log_file_location):
                os.makedirs(log_file_location)
            # 再次尝试写文件
            if not os.path.exists(file_path):
                with open(file_path, mode='w', encoding='utf-8') as f:
                    f.write(contents)
                    pass
        except Exception as e:
            print(e)
            raise


# 单个功能函数测试
if __name__ == '__main__':
    # 示例函数
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

    # 定义函数列表
    functions_list = [calculate_total_age_function]

    # 测试: 自动编写函数的 JSON Schema 效果
    generator = AutoFunctionGenerator(functions_list)
    function_descriptions = generator.auto_generate()
    print(function_descriptions)

# 多个功能函数测试
if __name__ == '__main__' and 0:
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
    # 定义输出路径
    output_path = os.path.join('.', 'function_describe.json')

    # 测试: 自动编写多个函数的 JSON Schema 效果
    generator = AutoFunctionGenerator(functions_list, output_path=output_path)
    function_descriptions = generator.auto_generate()
    print(function_descriptions)
    """ function_descriptions 输出示例(已格式化): 
    [
        {
            'name': 'calculate_total_age_function',
            'description': "计算年龄总和的函数，从给定的JSON格式字符串（按'split'方向排列）中解析出DataFrame，计算所有人的年龄总和，并以JSON格式返回结果。",
            'parameters': {
                'type': 'object',
                'properties': {
                    'input_json': {
                        'description': '表示含有个体年龄数据的JSON格式字符串',
                        'type': 'string'
                    }
                },
                'required': [
                    'input_json'
                ]
            },
            'returns': {
                'type': 'string',
                'description': '所有人的年龄总和，以JSON格式返回。'
            }
        },
        {
            'name': 'calculate_married_count',
            'description': '从给定的JSON格式字符串中解析出DataFrame，计算结婚人数，并以JSON格式返回结果。',
            'parameters': {
                'type': 'object',
                'properties': {
                    'input_json': {
                        'description': '包含个体数据（其中包括婚姻状态）的JSON格式字符串',
                        'type': 'string'
                    }
                },
                'required': [
                    'input_json'
                ]
            },
            'returns': {
                'type': 'string',
                'description': '结婚人数，以JSON格式返回'
            }
        }
    ]
    """
