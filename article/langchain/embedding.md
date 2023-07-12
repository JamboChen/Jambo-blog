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

然后将文本和先前创建的 embedding 模型传入 `Redis.from_documents` 方法中，这样就可以将文本转换为向量并储存到 Redis 中了。而数据在 Redis 中的 key 会以 `dos:<index_name>:<hash>` 的形式储存。

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

```text
[Document(page_content='（六）有下列情形之一的，乙方不得单方面解除项目聘用合同： 1．在承担国家重大科研项目期间； 2．掌握重大科技成果关键技术和资料，未脱离保密期； 3．被审查期间或因经济问题未作结案处理之前。 （七）本合同订立时所依据的客观情况发 生重大变化，致使合同无法履行，\n经甲乙双方协商不能就变更合同达成协议的，双方均可以单方面解除本合同。 \n（八）有下列情形之一的，甲方应当根据乙方在本单位的实际工作年限向其\n支付经济补偿： \n1．甲方提出解除本合同，乙方同意解除的； 2．乙方患病或者非因工负伤，医疗期满 后，不能从事原工作也不能从事由\n甲方安排的其他工作，甲方单方面解除本合同的； \n3．乙方年度考核或者聘期考核不合格， 又不同意甲方调整其工作岗位的，', metadata={'source': './contract.pdf', 'page': 7}),
 Document(page_content='（二）乙方有下列情形之一的，甲方可以随时单方面解除本合同： 1．在试用期内被证明不符合本岗位要求； 2．连续旷工超过 10 个工作日或者 1 年内累计旷工超过 20 个工作日的； 3．乙方公派出国或因私出境逾期不归；   \n4．违反工作规定或者操作规程，发生责 任事故，或者失职、渎职，造成严\n重后果的； \n5．严重扰乱工作秩序，致使甲方、其他单位工作不能正常进行的； 6．被判处拘役、有期徒刑缓刑以及有期 徒刑以上刑罚收监执行，或者被劳\n动教养的。 \n（三）乙方有下列情形之一，甲方可以单方面解除本合同，但是应当提前\n30 日以书面形式通知乙方： \n1．乙方患病或者非因工负伤，医疗期满 后，不能从事原工作也不能从事由\n甲方安排的其他工作的；', metadata={'source': './contract.pdf', 'page': 6}),
 Document(page_content='83．因工负伤，治疗终结后经劳动能力鉴定机构鉴定为 1 至 4 级丧失劳动能\n力的； \n4．患职业病以及现有医疗条件下难以治愈的严重疾病或者精神病的； 5．乙方正在接受纪律审查尚未作出结论的； 6．属于国家规定的不得解除本合同的其他情形的。 （五）有下列情形之一的，乙方可以随时单方面解除本合同： 1．在试用期内的； 2．考入普通高等院校的； 3．被录用或者选调为公务员的； 4．依法服兵役的。 除上述情形外，乙方提出解除本合同未能与甲方协商一致的，乙方应当坚持\n正常工作，继续履行本合同；6 个月后再次提出解除本合同仍未能与甲方协商一致的，即可单方面解除本合同。', metadata={'source': './contract.pdf', 'page': 7}),
 Document(page_content='7（一）甲乙双方协商一致，可以变更本合同的相关内容。 \n（二）本合同订立时所依据的法律、法规、规章和政策已经发生变化的，应\n当依法变更本合同的相关内容。 \n（三）本合同确需变更的，由甲乙双方按照规定程序签订《项目聘用合同变\n更书》（附后），以书面形式确定合同变更的内容。 \n（四）乙方年度考核或者聘期考核不合格，甲方可以调整乙方的岗位或安排\n其离岗接受必要的培训后调整岗位,并向乙方出具《岗位调整通知书》 （附后）,\n对本合同作出相应的变更。 （此款只适用于乙方属于甲方单位事业编制内职工。 ） \n七、项目聘用合同的解除  \n（一）甲方、乙方双方经协商一致，可以解除本合同。', metadata={'source': './contract.pdf', 'page': 6})]
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

```text
[Document(page_content='（六）有下列情形之一的，乙方不得单方面解除项目聘用合同： 1．在承担国家重大科研项目期间； 2．掌握重大科技成果关键技术和资料，未脱离保密期； 3．被审查期间或因经济问题未作结案处理之前。 （七）本合同订立时所依据的客观情况发 生重大变化，致使合同无法履行，\n经甲乙双方协商不能就变更合同达成协议的，双方均可以单方面解除本合同。 \n（八）有下列情形之一的，甲方应当根据乙方在本单位的实际工作年限向其\n支付经济补偿： \n1．甲方提出解除本合同，乙方同意解除的； 2．乙方患病或者非因工负伤，医疗期满 后，不能从事原工作也不能从事由\n甲方安排的其他工作，甲方单方面解除本合同的； \n3．乙方年度考核或者聘期考核不合格， 又不同意甲方调整其工作岗位的，', metadata={'source': './contract.pdf', 'page': 7}),
 Document(page_content='（二）乙方有下列情形之一的，甲方可以随时单方面解除本合同： 1．在试用期内被证明不符合本岗位要求； 2．连续旷工超过 10 个工作日或者 1 年内累计旷工超过 20 个工作日的； 3．乙方公派出国或因私出境逾期不归；   \n4．违反工作规定或者操作规程，发生责 任事故，或者失职、渎职，造成严\n重后果的； \n5．严重扰乱工作秩序，致使甲方、其他单位工作不能正常进行的； 6．被判处拘役、有期徒刑缓刑以及有期 徒刑以上刑罚收监执行，或者被劳\n动教养的。 \n（三）乙方有下列情形之一，甲方可以单方面解除本合同，但是应当提前\n30 日以书面形式通知乙方： \n1．乙方患病或者非因工负伤，医疗期满 后，不能从事原工作也不能从事由\n甲方安排的其他工作的；', metadata={'source': './contract.pdf', 'page': 6})]
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

# 与 Chain 结合

我们在 LangChain 基础那篇教程中介绍了一个基础的 Chain，它只能拼接 prompt，无法进行搜索，这对于我们现在的需求是不够的。针对需要搜索的场景，LangChain 提供了 `RetrievalQA` 类，它可以把检索器和 LLM 结合起来，这样就可以在 Chain 模型中进行搜索了。

与文档做交互的 Chain ，会有一个 `chain_type` 参数，它是用来指定处理文档的方式。比如 `map_reduce` 是把每个检索器返回的文档都分别用 llm 进行处理（map 过程），然后把处理结果和提出的问题拼接起来再让 llm 回答问题；而默认的 `stuff` 是直接把返回的文档直接拼接起来。具体可以参考[官方文档](https://python.langchain.com/docs/modules/chains/document/)。

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever)

print(qa.run("在什么情况下可以单方面解除合同"))
```

```text
根据所提供的文本，有以下情况可以单方面解除合同：

- 甲方因工负伤，治疗终结后经劳动能力鉴定机构鉴定为 1 至 4 级丧失劳动能力；
- 甲方患职业病以及现有医疗条件下难以治愈的严重疾病或者精神病；
- 甲方被判刑；
- 甲方违反劳动纪律、规章制度，给用人单位造成重大损害；
- 甲方正在接受纪律审查尚未作出结论；
- 甲方属于国家规定的不得解除本合同的其他情形；
- 乙方在试用期内；
- 乙方考入普通高等院校；
- 乙方被录用或者选调为公务员；
- 乙方依法服兵役。

除上述情形外，所有的变更和解除都需要甲乙双方协商一致。
```