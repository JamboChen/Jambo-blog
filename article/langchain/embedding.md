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

