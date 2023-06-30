# 文本提取

我们的目标是做出一个根据资料回答问题的机器人，那么从资料源中提取文本信息就是一件必要的事。但我们的资料源格式是多样的，比如 PDF、Word、HTML、PPT 等等，甚至有的资料源来自于网络，这些格式都不能直接提取出文本，但好在 Python 有很多第三方库可以帮助我们提取文本信息，并且 LangChain 也帮我们整合到了一起，我们只需要调用 LangChain 的接口就可以了。这里我们以 PDF 为例，介绍一下如何提取文本信息。

LangChain 针对 PDF 包含了许多第三方库，比如 `PyPDF2`、`PyPDFium2`、`PDFMiner` 等等，这里我们以 `PyPDF2` 为例，介绍一下如何提取文本信息。

```bash
pip install pypdf
```

我们使用 `PyPDFLoader` 来加载 PDF 文件，然后调用 `load` 方法就可以得到文本信息了。PDF 的读取器会将 PDF 文件中的每一页转换成一段文本，然后将这些文本段组成一个列表返回。

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("./contract.pdf")
documents = loader.load()
```

但仅仅是按页进行分割，仍然是不够的，因为一页中仍然包含了太多的字数，如果将来将这些文本放在 prompt 的上下文中，会占据大量的 token，而这些文本对于当下所问的问题不一定都是有用的，所以我们需要将这些文本再进行分割。

LangChain 提供了几种分割器，可以分割代码、段落、Markdown 标题。这里我们使用 `RecursiveCharacterTextSplitter` ，他会把文本按照字符进行分割，直到每个文本段的长度足够小。它默认的分割列表是 `["\n\n", "\n", " ", ""]`，这样它可以尽可能把段落、句子或单词放在一起。

其中 `chunk_size` 限制了每段文本的最大长度。`chunk_overlap` 则是两个文本段之间的重叠长度，如果一个段落实在太长而被强制分割成了两段，那么这两段之间就会有一段重叠的文本，这样可以保证上下文的连贯性。

它还有个参数 `length_function`，用来计算文本的长度，默认是 `len`。如果你想按照 token 的数量来分割，那么可以结合 `tiktoken` 库来使用。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
```