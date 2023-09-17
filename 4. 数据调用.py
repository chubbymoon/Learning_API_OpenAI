import types
import pandas as pd
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# 创建一个稍微复杂的 DataFrame，包含多种数据类
df_complex = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000.0, 100000.5, 150000.75],
    'IsMarried': [True, False, True]
})

df_complex_json = df_complex.to_json(orient='split')

# print(df_complex_json)
# {"columns":["Name","Age","Salary","IsMarried"],"index":[0,1,2],"data":[["Alice",25,50000.0,true],["Bob",30,100000.5,false],["Charlie",35,150000.75,true]]}


response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo-0613",
    messages = [
        {"role": "system", "content": f"你是一位优秀的数据分析师，现在有这样一份数据集：{df_complex_json}"},
        {"role": "user", "content": "请解释一下这个数据集的分布情况"}
    ],
    stream = True
)


# 处理 "流式输出" 或 整体输出
if type(response) == types.GeneratorType:
    for i in response:
        try:
            # print(i, end="")
            print(i.choices[0].delta['content'], end="")
        except KeyError:
            pass
else:
    # print(response)
    print(response.choices[0].message['content'])



