在上个教程中，我们从资料源中获取了文本信息，但我们还没有将其储存起来或者做任何其他处理，而这就是我们接下来要做的事情。

根据输入的问题匹配合适的资料条目是件非常困难的事情，毕竟这相当于是要手搓一个搜索引擎。虽然也有像 Azure 认知搜索之类的服务，但是如果你要在本地进行储存以及搜索，更好的方法无疑是使用 embedding。

Embedding 是一种将文本转换为向量的方法，这样我们就可以使用向量的相似度来判断两个文本的相似度。而 Azure OpenAI 也提供了 embedding 模型，并且 LangChain 也将其整合到了一起，初次之外还有其他各种储存向量的数据库，比如 Milvus、Faiss 等等，但这里我们将会用 Redis 作为示范。

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

```bash

```

我们按照上次教程里的方法，从 PDF 中提取文本信息，然后和刚刚创建的 embedding 模型一起使用，将文本转换为向量，然后储存起来。

```python
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./contract.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

rds = Redis.from_documents(
    docs, embeddings, redis_url="redis://localhost:6379", index_name="link"
)
```

