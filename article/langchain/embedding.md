在上个教程中，我们从资料源中获取了文本信息，但我们还没有将其储存起来或者做任何其他处理，而这就是我们接下来要做的事情。

根据输入的问题匹配合适的资料条目是件非常困难的事情，毕竟这相当于是要手搓一个搜索引擎。虽然也有像 Azure 认知搜索之类的服务，但是如果你要在本地进行储存以及搜索，更好的方法无疑是使用 embedding。

Embedding 是一种将文本转换为向量的方法，这样我们就可以使用向量的相似度来判断两个文本的相似度。

# Embedding

和之前一样，我们要先设置环境变量

```python
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
```

然后实例化 embedding 模型对象。这里我们除了要指定模型的部署名，还要设定 `chunk_size=1`，因为我的模型一次只能处理一条文本，如果多于一条文本，就会报错。

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1)
```

如果要直接使用 embedding 模型，可以使用 `embed_query` 方法将文本转换为向量

```python
text = "This is a test document."

query_result = embeddings.embed_query(text)
```

或者用 `embed_documents` 批量将文本转换为向量

```python
doc_result = embeddings.embed_documents([text])
```

# 储存向量

在我们将文本转换成向量后，我们还要将其储存起来，这样我们才能在之后的搜索中使用。LangChain 整合了许多储存向量的数据库，这里我们用 Redis 作为例子。具体来说，我这里的 Redis 是在 WSL 上通过 Docker 运行的 RediSearch。

在使用 Redis 储存向量前，我们首先需要安装 Python 的 Redis 库。

```bash
pip install redis
```

我们按照上次教程里的方法，从 PDF 中提取文本信息

```python
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./contract.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
```

然后将文本和先前创建的 embedding 模型传入 `Redis.from_documents` 方法中，这样就可以将文本转换为向量并储存到 Redis 中了。

```python
rds = Redis.from_documents(
    docs,
    embeddings,
    redis_url="redis://localhost:6379",
    index_name="link",
)
```

我们将资料放入数据库后，就可以使用 `similarity_search` 方法匹配相似的文本了。它会把输入的查询也转换为向量，然后调用数据库的搜索方法，返回最相似的前 4 个文本。

```python
rds.similarity_search("在什么情况下可以单方面解除合同")
```

# 检索器

显然，我们可以很容易的操作向量数据库用相似度方法来匹配文本，但这仍然过于简单，并且在语义上也只是从数据库调取数据，而不是真正的搜索。而 LangChain 也提供了检索器（Retriever）的功能，并且我们可以把现有的向量数据库转换为检索器。

```python
retriever = rds.as_retriever()
```

这样我们就拿到了一个使用 Redis 作为后端的检索器，要进行搜索，只需要调用 `get_relevant_documents` 方法，最后的结果和之前是一样的。

```python
retriever.get_relevant_documents("在什么情况下可以单方面解除合同")
```

当然，你也可以在转换为检索器时设定搜索的参数，比如这里设置 `k=2` ，最多就只会返回 2 个文本。并且搜索方式设为 `similarity_limit`，相应的，我们设定 `threshold=0.8` 来限制相似度的阈值，这样就不会返回相似度低于 0.8 的文本了。

```python
retriever = rds.as_retriever(search_type="similarity_limit", k=2, score_threshold=0.8)

retriever.get_relevant_documents("在什么情况下可以单方面解除合同")
```

要注意的是，不同的数据库转换为检索器时，支持的参数不一定相同，具体可以查看[官方文档](https://python.langchain.com/docs/modules/data_connection/vectorstores/)，或者源码。

上面所介绍的是基于向量数据库的搜索方式，但 LangChain 还集成了 Azure 认知搜索、SVM 等搜索方式，具体可以参考[文档](https://python.langchain.com/docs/modules/data_connection/retrievers/integrations/azure_cognitive_search)，这里同样不再赘述。

## 用 LLM 加强检索器

不管是基于 embedding 模型的向量数据库还是其他算法，本质上都是根据“距离”来计算相似度。但在现实中，当查询语句的措辞稍微发生细微的改变，那么就可能导致搜索结果的巨大变化。这时就需要手动调整来解决，但这通常是一件非常耗时的事情，并且我们无法对所有的查询语句都手动进行调整。

这时我们就可以引入 LLm 来自动进行这一过程。LangChain 已经把 prompt 编写好，把这一整个过程封装成了一个类，我们只需要传入之前创建的检索器，和 LLM 的实例。

```python
from langchain.chat_models import AzureChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
```

我们设置了一个 logger，这样在运行的过程中可以看到它究竟生成了什么样的查询语句。

```python
import logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
```

它会根据你的查询语句，生成 3 个不同的查询语句。然后用这 3 个查询语句去检索，最后把结果合并起来，返回最终的结果。

```python
unique_docs = retriever_from_llm.get_relevant_documents("在什么情况下可以单方面解除合同")

len(unique_docs)
```

```text
INFO:langchain.retrievers.multi_query:Generated queries: ['1. 什么条件下可以单方面终止合同？', '2. 哪些情况下可以单方面解除合同？', '3. 在哪些情况下可以单方面取消合同？']

2
```

除此之外，还有利用 LLM 像对输出的文本进行压缩，或者是格式化查询之类的，限于篇幅，这里就不再赘述了，大家可以参考[官方文档](https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/contextual_compression/)。