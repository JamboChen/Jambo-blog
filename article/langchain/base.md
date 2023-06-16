title: LangChain

# 为什么要用 LangChain

许多开发者希望将像 GPT 这样的大语言模型整合到他们的应用中。而这些应用不仅仅是简单地将用户的输入传递给 GPT，然后将 GPT 的输出返回给用户。

这些应用可能需要根据特定的资料源来回答问题，因此需要考虑如何存储和查找资料。或者需要整理用户的输入，保存以前的消息记录并提取重点。如果你希望模型按照特定的格式输出文本，那么你需要在 prompt（提示）中详细描述格式，甚至需要提供示例。这些 prompt 通常是应用程序后台进行管理，用户往往不会注意到它们的存在。对于一些复杂的应用程序，一个问题可能需要多个执行动作。例如声称可以自动完成指定项目的 AutoGPT，实际上是根据目标和作者编写的 prompt 生成所需的动作并以JSON格式输出，然后程序再执行相应的动作。

LangChain 基本上已经将这些你可能会使用到的功能打包好了，只需要规划程式逻辑并调用函数即可。此外，LangChain 的这些功能与具体使用的模型API无关，不必为不同的语言模型编写不同的代码，只需更换 API 即可。

# 基本用法

在使用 LangChain 之前，建议先了解 OpenAI API 的调用，否则即使是使用 LangChain，参数和用法也可能不容易理解。具体可以参考我之前的教程：<>

下面我们会使用 Azure OpenAI API 作为演示。另外，LangChain 将由文字续写（补全）的语言模型称为 llm ，拥有聊天界面（输入为聊天记录）的语言模型称为聊天模型。

## 安装

因为 LangChain 在调用 OpenAI 的 API 时，实际上会使用 OpenAI 提供的 SDK，因此我们还需要一并安装 `openai` 。

```
pip install langchain
pip install openai
```

## 生成文本

### 实例化模型对象

在使用 OpenAI API 之前，我们需要先设置环境变量。如果你使用的是 OpenAI 原生的接口，就只需要设置 `api_key`；如果是 Azure 则还需要设置 `api_version` 和 `api_base` ，具体的值与使用 `openai` 库调用 azure 接口一样，可以参考我之前的教程：<>

```python
import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_VERSION"] = ""
os.environ["OPENAI_API_BASE"] = ""
```

当然，这些值也可以在 terminal 中使用 `export` （在 Linux 下）命令设置；或者在 .env 文件中设置，然后用 `python-dotenv` 库导入进环境变量。

LangChain 的大语言模型（llm）的类都封装在 `llms` 中，我们需要从中导入 `AzureOpenAI` 类，并设置相关的参数。其中指定模型的参数名是 `deployment_name`，剩下的参数就是 OpenAI API 的参数了。事实上，上面在环境变量中设置的 API 信息也可以在这里作为参数传入，但考虑到便利性和安全性，仍建议在环境变量中设置 API 信息。

要注意的是，prompt 和 stop 参数并不是在这里传入的，而是在下面生成文本时传入。

```python
from langchain.llms import AzureOpenAI
llm = AzureOpenAI(
    deployment_name="text-davinci-003",
    temperature=0.9,
    max_tokens=265,
)
```

另外，如果你使用的是原生 OpenAI API ，那么导入的类应该是 `OpenAI` ，并且指定模型的参数名是 `model_name`，例如：

```python
from langchain.llms import AzureOpenAI
llm = AzureOpenAI(model_name="text-davinci-003")
```

#### 序列化 LLM 配置

假如你需要对多个场景有不同的 llm 配置，那么将配置写在代码中就会不那么简单灵活。在这种情况下，将 llm 配置保存在文件中显然会更方便。

```python
from langchain.llms import OpenAI
from langchain.llms.loading import load_llm
```

LangChain 支持将 llm 配置以 json 或 yaml 的格式读取或保存。假设我现在有一个 `llm.json` 文件，内容如下：

```json
{
    "model_name": "text-davinci-003",
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "n": 1,
    "best_of": 1,
    "request_timeout": null,
    "_type": "openai"
}
```

那么我们可以使用 `load_llm` 函数将其转换成 llm 对象，具体使用的是什么语言模型是使用 `_type` 参数定义。

```python
llm = load_llm("llm.json")
# llm = load_llm("llm.yaml")
```

当然你也可以从 llm 对象导出成配置文件。
```
llm.save("llm.json")
llm.save("llm.yaml")
```

### 从文本生成文本

接下来我们就要使用上面实例化的模型对象来生成文本。LangChain 的 llm 类有三种方法从 String 生成文本：`predict()` 方法、`generate()`方法、像函数一样直接调用对象（`__call__`）。

看上去途径很多，但其实都只是 `generate()` 一种而已。具体来说，`perdict()` 简单检查后调用了 `__call__` ，而 `__call__` 简单检查后调用了 `generate()`。`generate()` 方法与其他两种途径最大的区别在于 prompt 的传入和返回的内容：`generate()` 传入的是包含 prompt 的 list 并返回一个 `LLMResult` 对象，而其他两种方法传入的是 prompt 本身的 string ，返回生成文本的 string。意思是 `generate()` 可以一次对多个 prompt **独立**生成对应的文本。

```python
prompt = "1 + 1 = "
stop = ["="]
# 下面三种生成方法是等价的
res1 = llm(prompt, stop=stop)
res2 = llm.predict(prompt, stop=stop)
res3 = llm.generate([prompt], stop=stop).generations[0][0].text
```

如果只是想单纯的从文字续写（生成）文字的话，推荐用 `predict()` 方法，因为这种最方便也最直观。

## 聊天模型

### 实例化模型对象

与上面的生成模型一样，我们需要先设置环境变量

```python
import os
os.environ["OPENAI_API_KEY"] = ""
```

LangChain 的聊天模型包装在 `langchain.chat_models` 下。我们对聊天模型输入的 prompt 不再是文字，而是消息记录，消息记录中是用户和模型轮流对话的内容，这些消息被 LangChain 包装为 `AIMessage`、`HumanMessage`、`SystemMessage`，其中的 `SystemMessage` 可以理解为给模型设置的人设。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
```

我们先构建一个初始的消息记录，当然 `SystemMessage` 并不是必须的。

```python
messages = [
    SystemMessage(content="你是一名翻译员，将中文翻译成英文"),
    HumanMessage(content="你好世界")
]
chat(messages)
```

和之前一样，聊天模型的 `generate()` 方法也支持对多个聊天记录生成消息

```python
result = chat.generate([messages, messages])
```

## LLMResult

上面说 llm 的 `generate()` 方法返回的是一个 `LLMResult` 对象，它由三个部分组成：`generations` 储存生成的文字和对应的信息、`llm_output` 储存 token 使用量和使用的模型、`run` 储存了唯一的 `run_id`，这是为了方便在生成过程中调用回调函数。通常我只需要关注 `generations` 和 `llm_output` 。

为了展示 `LLMResult` 的结果，这里我们重新创建一个 llm 对象，并设置参数 `n=2` ，代表模型对于每个 prompt 会生成两次结果，这个值默认是 `n=1`。

```python
llm = AzureOpenAI(deployment_name="text-davinci-003", temperature=0, n=2)
llm_result = llm.generate([f"{i}**2 =" for i in range(1, 11)], stop="\n")
print(len(llm_result.generations))
# -> 10
print(len(llm_result.generations[0]))
# -> 2
```


因为 `LLMResult` 是继承自 Pydantic 的 `BaseModel` ，因此可以用 `json()` 将其格式化为 JSON ：

```python
print(llm_result.json())
```

```json
{
    "generations": [
        [
            {
                "text": " 1",
                "generation_info": {
                    "finish_reason": "stop",
                    "logprobs": null
                }
            },
            {
                "text": " 1",
                "generation_info": {
                    "finish_reason": "stop",
                    "logprobs": null
                }
            }
        ],
        ...
    ],
    "llm_output": {
        "token_usage": {
            "prompt_tokens": 40,
            "total_tokens": 58,
            "completion_tokens": 18
        },
        "model_name": "text-davinci-003"
    },
    "run": {
        "run_id": "cf7fefb2-2e44-474d-918f-b8695a514646"
    }
}
```

可以看到 `generations` 列表储存了模型为对应 prompt 生成的内容，由之前的代码可知一它的长度为 10 。因为设置了 `n=2` 所以生成内容的 list 中包含了两个生成文本信息。


## Prompt 模板

很多时候我们并不会把用户的输入直接丢给模型，可能会需要在前后文进行补充信息，而这个补充的信息就是“模板”。下面是一个简单的例子，这个 prompt 包含一个输入变量 `product`，：

```python
template = """
我希望你担任顾问，帮忙为公司想名字。
这个公司生产{product}，有什么好名字？
"""
```

我们可以用 `PromptTemplate` 将这个带有输入变量的 prompt 包装成一个模板。

```python
from langchain import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["product"],
    template=template,
)
prompt_template.format(product="运动衬衫")
# -> 我希望你担任顾问，帮忙为公司想名字。
# -> 这个公司生成运动衬衫，有什么好名字？
```

当然，如果 prompt 中没有输入变量，也可以将其用 `PromptTemplate` 包装，只是 `input_variables` 参数输入的是空列表。

如果你不想手动指定 `input_variables` ，也可以使用 `from_template()` 方法自动推导。

```python
prompt_template = PromptTemplate.from_template(template)
prompt_template.input_variables
# -> ['product']
```

你还可以将模板保存到本地文件中，目前 LangChain 只支持 json 和 yaml 格式，它可以通过文件后缀自动判断文件格式。

```python
prompt_template.save("awesome_prompt.json") # 保存为 json
```

也可以从文件中读取

```python
from langchain.prompts import load_prompt
prompt_template = load_prompt("prompt.json")
```

## Chain 

Chain 是 LangChain 里非常重要的概念（毕竟都放在名字里了），它与管道（Pipeline）类似，就是将多个操作组装成一个函数（流水线），从而使得代码更加简洁方便。

比如我们执行一个完整的任务周期，需要先生成 prompt ，将 prompt 给 llm 生成文字，然后可能还要对生成的文字进行其他处理。更进一步，我们或许还要记录任务每个阶段的日志，或者更新其他数据。这些操作如果都写出来，那么代码会非常冗长，而且不容易复用，但是如果使用 Chain ，你就可以将这些工作都打包起来，并且代码逻辑也更清晰。你还可以将多个 Chain 组合成一个更复杂的 Chain 。

我们首先创建一个 llm 对象和一个 prompt 模板。

```python
from langchain.llms import AzureOpenAI
from langchain import PromptTemplate

llm = AzureOpenAI(deployment_name="text-davinci-003", temperature=0)
prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    将给定的字符串进行大小写转换。
    例如：
    输入： ABCdef
    输出： abcDEF
    
    输入： AbcDeF
    输出： aBCdEf
    
    输入： {input}
    输出： 
    """,
)
```

接下来我们可以通过 `LLMChain` 将 llm 和 prompt 组合成一个 Chain 。这个 Chain 可以接受用户输入，然后将输入填入 prompt 中，最后将 prompt 交给 llm 生成结果。另外如果你用的是聊天模型，那么使用的 Chain 是 `ConversationChain`。

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("HeLLo"))
# -> hEllO
```

如果 prompt 中有多个输入变量，可以使用字典一次将它们传入。

```python
print(chain.run({"input": "HeLLo"}))
# -> hEllO
```

### Debug 模式

上面都是些简单的例子，只牵扯到少量的输入变量，但在实际使用中可能会有大量的输入变量，并且 llm 的输出还是不固定的，这就使得我们很难从最重的结果反推问题所在。为了解决这个问题，LangChain 提供了 `verbose` 模式，它会将每个阶段的输入输出都打印出来，这样就可以很方便地找到问题所在。

```python
chain_verbose = LLMChain(llm=llm, prompt=prompt, verbose=True)
print(chain_verbose.run({"input": "HeLLo"}))
```

```text
> Entering new  chain...
Prompt after formatting:

    将给定的字符串进行大小写转换。
    例如：
    输入： ABCdef
    输出： abcDEF
    
    输入： AbcDeF
    输出： aBCdEf
    
    输入： HeLLo
    输出： 
    

> Finished chain.
 hEllO
```

### 组合 Chain

一个 Chain 对象只能完成一个很简单的任务，但我们可以像搭积木一样将多个简单的动作组合在一起，就可以完成更复杂的任务。其中最简单的是顺序链 `SequentialChain`，它将多个 Chain 串联起来，前一个 Chain 的输出将作为后一个 Chain 的输入。不过要注意的是，因为这只是个最简单的链，所以它不会对输入进行任何处理，也不会对输出进行任何处理，所以你需要保证每个 Chain 的输入输出都是兼容的，并且它要求每个 Chain 的 prompt 都只有一个输入变量。


下面，我们先计算输入数字的平方，然后将平方数转换成罗马数字。

```python
from langchain.chains import SimpleSequentialChain

llm = AzureOpenAI(deployment_name="text-davinci-003", temperature=0)
prompt1 = PromptTemplate(
    input_variables=["base"], template="{base}的平方是： "
)
chain1 = LLMChain(llm=llm, prompt=prompt1)

prompt2 = PromptTemplate(input_variables=["input"], template="将{input}写成罗马数字是： ")
chain2 = LLMChain(llm=llm, prompt=prompt2)

overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
overall_chain.run(3)

```

```text
> Entering new  chain...


9
IX

> Finished chain.
'IX'
```

LangChain 已经预先准备了许多不同 Chain 的组合，具体可以参考官方文档，这里就先不展开了。