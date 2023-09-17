import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# 通过拼接 上一次的回复进行多轮对话
completion = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo-0613",
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": "你是一位精通机器学习和自然语言处理的 AI 领域专家，具备 20 年相关经验"},

              {"role": "user", "content": "我是一个小白，我想入门 AI 领域，我需要学习哪些知识"},

              {"role": "assistant", "content": "\u4f5c\u4e3a\u4e00\u4e2a AI \u9886\u57df\u7684\u5c0f\u767d\uff0c\u4f60\u53ef\u4ee5\u901a\u8fc7\u4ee5\u4e0b\u6b65\u9aa4\u6765\u5165\u95e8\uff1a\n\n1. \u5b66\u4e60\u673a\u5668\u5b66\u4e60\u57fa\u7840\u77e5\u8bc6\uff1a\u673a\u5668\u5b66\u4e60\u662f AI \u9886\u57df\u7684\u6838\u5fc3\u5185\u5bb9\uff0c\u4f60\u9700\u8981\u4e86\u89e3\u7edf\u8ba1\u5b66\u3001\u7ebf\u6027\u4ee3\u6570\u548c\u5fae\u79ef\u5206\u7b49\u57fa\u7840\u6570\u5b66\u6982\u5ff5\u3002\u4e86\u89e3\u5e38\u89c1\u7684\u673a\u5668\u5b66\u4e60\u7b97\u6cd5\uff0c\u5982\u7ebf\u6027\u56de\u5f52\u3001\u51b3\u7b56\u6811\u3001\u652f\u6301\u5411\u91cf\u673a\u7b49\u3002\n\n2. \u5b66\u4e60\u7f16\u7a0b\u548c\u6570\u636e\u5904\u7406\uff1aAI \u9886\u57df\u79bb\u4e0d\u5f00\u7f16\u7a0b\uff0cPython \u662f\u4e00\u95e8\u5e38\u7528\u7684\u7f16\u7a0b\u8bed\u8a00\uff0c\u4f60\u53ef\u4ee5\u5b66\u4e60 Python \u5e76\u638c\u63e1\u4e00\u4e9b\u6570\u636e\u5904\u7406\u548c\u5206\u6790\u7684\u5e93\uff0c\u5982 NumPy\u3001Pandas \u548c Matplotlib\u3002\n\n3. \u5b66\u4e60\u6df1\u5ea6\u5b66\u4e60\uff1a\u6df1\u5ea6\u5b66\u4e60\u662f AI \u9886\u57df\u7684\u91cd\u8981\u5206\u652f\uff0c\u4f60\u53ef\u4ee5\u5b66\u4e60\u6df1\u5ea6\u795e\u7ecf\u7f51\u7edc\u7684\u539f\u7406\u548c\u5e38\u7528\u7684\u6df1\u5ea6\u5b66\u4e60\u6846\u67b6\uff0c\u5982 TensorFlow \u6216 PyTorch\u3002\n\n4. \u5b66\u4e60\u81ea\u7136\u8bed\u8a00\u5904\u7406\uff1a\u81ea\u7136\u8bed\u8a00\u5904\u7406\u662f AI \u9886\u57df\u7684\u4e00\u4e2a\u91cd\u8981\u5e94\u7528\u65b9\u5411\uff0c\u4f60\u53ef\u4ee5\u5b66\u4e60\u6587\u672c\u5904\u7406\u7684\u57fa\u7840\u6280\u672f\uff0c\u5982\u5206\u8bcd\u3001\u8bcd\u6027\u6807\u6ce8\u3001\u547d\u540d\u5b9e\u4f53\u8bc6\u522b\u7b49\u3002\u4e86\u89e3\u5e38\u89c1\u7684\u81ea\u7136\u8bed\u8a00\u5904\u7406\u6a21\u578b\uff0c\u5982\u8bcd\u5d4c\u5165\u3001\u5faa\u73af\u795e\u7ecf\u7f51\u7edc\u548c\u6ce8\u610f\u529b\u673a\u5236\u3002\n\n5. \u5b9e\u8df5\u9879\u76ee\uff1a\u901a\u8fc7\u5b9e\u8df5\u9879\u76ee\u6765\u5de9\u56fa\u6240\u5b66\u77e5\u8bc6\uff0c\u53ef\u4ee5\u4ece\u516c\u5f00\u6570\u636e\u96c6\u5f00\u59cb\uff0c\u5982 MNIST \u624b\u5199\u6570\u5b57\u8bc6\u522b\u4efb\u52a1\uff0c\u7136\u540e\u9010\u6e10\u6311\u6218\u66f4\u590d\u6742\u7684\u95ee\u9898\uff0c\u5982\u56fe\u50cf\u5206\u7c7b\u3001\u6587\u672c\u751f\u6210\u7b49\u3002\n\n6. \u53c2\u4e0e\u793e\u533a\u548c\u5b66\u4e60\u8d44\u6e90\uff1a\u53c2\u4e0e AI \u793e\u533a\uff0c\u5982 Kaggle \u7ade\u8d5b\u3001GitHub \u5f00\u6e90\u9879\u76ee\u7b49\uff0c\u4e0e\u5176\u4ed6\u4ece\u4e1a\u8005\u4ea4\u6d41\u7ecf\u9a8c\u3002\u6b64\u5916\uff0c\u8fd8\u53ef\u4ee5\u5229\u7528\u5728\u7ebf\u8d44\u6e90\u548c\u6559\u7a0b\uff0c\u5982\u5404\u79cd AI \u76f8\u5173\u7684\u535a\u5ba2\u3001\u6559\u80b2\u5e73\u53f0\u548c MOOC\uff08\u5982 Coursera\u3001Udacity\u3001edX \u7b49\uff09\u3002\n\n\u9700\u8981\u6ce8\u610f\u7684\u662f\uff0cAI \u9886\u57df\u5341\u5206\u5e7f\u6cdb\u6df1\u5965\uff0c\u6d89\u53ca\u591a\u4e2a\u65b9\u9762\u7684\u77e5\u8bc6\u548c\u6280\u672f\uff0c\u521a\u5f00\u59cb\u4f1a\u611f\u5230\u6709\u4e9b\u56f0\u96be\uff0c\u4f46\u53ea\u8981\u4fdd\u6301\u5174\u8da3\u548c\u575a\u6301\u5b66\u4e60\uff0c\u4f60\u4e00\u5b9a\u53ef\u4ee5\u638c\u63e1 AI \u9886\u57df\u7684\u77e5\u8bc6\u548c\u6280\u80fd\u3002"},

              {"role": "user", "content": "关于第 3 条深度学习方面，你帮我更加详细的介绍一下"}

              ]
)

print(completion)
# print(completion.choices[0].message)

# TODO 回复为 Unicode 编码 要转为 可读文本(汉语),
# TODO 如何向聊天一样对话(自动拼接上下文)