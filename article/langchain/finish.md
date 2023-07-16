上一篇教程我们介绍了 ReAct 系统，这是一个非常强大的行为模式，但它需要编写大量的示例来告诉 LLM 如何思考、行动，并且为了遵循这个模式，还需要编写代码来分析生成文字、调用函数、拼接 prompt 等，这些工作都是十分繁琐且冗长的。而 LangChain 帮我们把这些步骤都隐藏起来了，将这一系列动作都封装成 “代理”，我们只需要提供有那些工具可以使用，以及这些工具的功能，就可以让 LLM 自动完成这些工作了。这样我们的代码就会变得简洁、易读，并且拥有更高的灵活性。

# 代理（Agent）

所谓代理，就是将一系列的行为和可以使用的工具封装起来，让我们可以通过一个简单的接口来调用这些动作，而不需要关心这些动作是如何完成的。这样可以让我们的代码更简洁、易读，并且拥有更高的灵活性。接下来我们就结合 ReAct 和向量数据库来介绍代理的使用。

## 准备工作

和之前一样，我们先设置环境变量。

```python
# set the environment variables needed for openai package to know to reach out to azure
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
```

接下来我们先把 llm 类创建好。

```python
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", temperature=0)
embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1)
```

因为我们的向量数据库已经在之前的教程中创建好了，所以我们用 `from_existing_index` 直接连接到数据库。这里 `index_name` 是我们之前创建数据库时指定的名字。另外不同的数据库所使用的链接方法可能不同，具体还是要参考各个数据库的[文档](https://python.langchain.com/docs/modules/data_connection/vectorstores/)。

```python
from langchain.vectorstores.redis import Redis

rds = Redis.from_existing_index(
    embeddings, redis_url="redis://localhost:6379", index_name="link"
)
retriever = rds.as_retriever()
```

## 创建工具

虽然代理可以让 llm 使用外部工具，但我们首先要创建工具，并为工具编写描述。我们的目的是为了要能够让 llm 根据问题从数据库中检索出答案，因此我们创建的工具一定是与此相关的。我们可以将检索器的搜索功能包装成一个函数，然后以此作为工具，但我们完全可以用之前教程中所介绍的 `RetrievalQA` Chain ，它的输出本就是字符串，而且已经先将相关的资料进行了整理，这显然比检索器更高效。

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
```

然后用 `Tool` 将上面创建的 Chain 包装起来。其中 `func` 参数是告诉代理系统要调用哪个函数，`description` 是对这个工具的描述。我们将包装好的工具放在一个列表中，这样就是告诉系统他可以使用的工具有哪些了。

```python
from langchain.agents import Tool

tools = [
    Tool(
        name="Contract QA System",
        func=qa.run,
        description="useful for when you need to answer questions about contracts."
    )
]
```

## 创建代理

我们用 `initialize_agent` 函数来创建代理。这里 `agent` 参数是告诉系统我们要创建的代理类型，这里我们创建的是 `CHAT_ZERO_SHOT_REACT_DESCRIPTION`，名字中的 `CHAT` 表示这是一个聊天代理，因为我们用的是聊天模型；`ZERO_SHOT` 表示它的 prompt 没有编写示例，而是仅通过描述来引导 llm 的行为。

```python
agent_chain = initialize_agent(
    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```

然后用 `run` 方法来运行代理。

```python
agent_chain.run("在什么情况下可以单方面解除合同？")
```

```text
> Entering new  chain...
Thought: This question is asking about the circumstances under which a contract can be unilaterally terminated.

Action:
\```
{
  "action": "Contract QA System",
  "action_input": "在什么情况下可以单方面解除合同？"
}
\```


Observation: 根据文本，可以单方面解除合同的情况包括：

- 在承担国家重大科研项目期间；
- 掌握重大科技成果关键技术和资料，未脱离保密期；
- 被审查期间或因经济问题未作结案处理之前；
- 合同订立时所依据的客观情况发生重大变化，致使合同无法履行，经甲乙双方协商不能就变更合同达成协议的；
- 甲方提出解除本合同，乙方同意解除的；
- 乙方患病或者非因工负伤，医疗期满后，不能从事原工作也不能从事由甲方安排的其他工作，甲方单方面解除本合同的；
- 乙方年度考核或者聘期考核不合格，又不同意甲方调整其工作岗位的；
- 在试用期内被证明不符合本岗位要求；
- 连续旷工超过 10 个工作日或者 1 年内累计旷工超过 20 个工作日的；
...

Final Answer: 根据文本，可以单方面解除合同的情况有很多种，包括在承担国家重大科研项目期间、合同订立时所依据的客观情况发生重大变化、甲方提出解除本合同且乙方同意解除等。此外，乙方在试用期内、考入普通高等院校、被录用或者选调为公务员、依法服兵役的情况下也可以随时单方面解除合同。如果乙方提出解除合同未能与甲方协商一致，乙方应当坚持正常工作，继续履行合同；6个月后再次提出解除合同仍未能与甲方协商一致的，即可单方面解除合同。

> Finished chain.

'根据文本，可以单方面解除合同的情况有很多种，包括在承担国家重大科研项目期间、合同订立时所依据的客观情况发生重大变化、甲方提出解除本合同且乙方同意解除等。此外，乙方在试用期内、考入普通高等院校、被录用或者选调为公务员、依法服兵役的情况下也可以随时单方面解除合同。如果乙方提出解除合同未能与甲方协商一致，乙方应当坚持正常工作，继续履行合同；6个月后再次提出解除合同仍未能与甲方协商一致的，即可单方面解除合同。'
```

这样我们很容易的完成 ReAct 的流程，并且它确实可以从资料源中检索出答案。用 LangChain 我们就可以用简洁的代码完成如此复杂的任务，并且也不需要编写 prompt，这样我们就可以更专注于业务的逻辑了。