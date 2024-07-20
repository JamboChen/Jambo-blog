# 用 Phi3-vision 进行简单线性回归

首先要说明的是，这并不是一篇回归分析的教程，而是通过回归分析来分享我对 Phi3-vision 在应用上的一些想法。尽管文中会涉及一些回归分析的理论，但这些并不是本文的重点，因此不会详细展开。请记住，本文的重心是在分享 Phi3-vision 的使用方法，即使不理解回归分析的理论也不会影响你阅读本文。

## 什么是线性回归

线性回归是一种分析、预测数据的一种方法。粗略的说，它试图用一条直线来描述 X 和 Y 之间的关系。

[![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1280px-Linear_regression.svg.png)](https://en.wikipedia.org/wiki/Regression_analysis)

在回归分析的理论中，当数据满足以下条件时，我们可以找到最好的那条直线：
- 残差（Y 与直线的距离）应该服从[常态分布](https://en.wikipedia.org/wiki/Normal_distribution)，即 Y 点应该均匀的分散在直线的两侧，并且不应该离太远，也不应该集中在一侧。
- 残差的方差应该是恒定的，即数据点分散程度应该是一致的，并不会因为任何因素而改变。

显然，这两个条件是非常理想的，在现实中很难完全满足。因此，回归分析中除了要寻找潜在的变量，还需要对数据进行变换，使其尽可能满足这两个条件。

### 例子

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/example.png)

从图中，我们可以很容易得出两个结论：
1. 随着 X 越大，Y 也呈现增大的趋势。
2. 左边的点比右边的点更集中。意思是，当 X 越大时，Y 的值的变化范围也越大。

这张图并不满足上面提到的条件，因此如果直接用线性回归方法去拟合这组数据，结果可能不理想。下图中，请关注 R-squared 值，这个值越接近 1，说明模型越好。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/lse.png)

通过一些变换方法，我们可以得到一个更好的模型。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/wlse.png)

当然，有很多变换方法，具体使用哪种变换，很多时候是凭借经验和直觉“看”出来的。

## 为什么我认为 Phi3-vision 适合做回归分析

虽然回归分析的理论非常严谨，但在实际应用中，往往需要主观判断。因为现实中的数据并不如理论那般完美，充满不可预知的变数，因此回归分析并不是在找数据间的因果关系，而是提供一个最可靠的方式去“猜测”数据。正如[“所有模型都是错误的”](https://en.wikipedia.org/wiki/All_models_are_wrong)，我们只是在寻找一个适合我们数据的模型。

尽管统计领域有很多理论去检验某个模型是否适合我们的数据，但过程中依然需要先看图表，再猜测数据的模型，然后验证这个数据模型是否合适。这其中也包含了一些看图说故事的部分。而 Phi3-vision 能够快速地根据图表给我们一些“主观”的判断。

## 回归分析的流程

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/flow.png)

这是一个简化的流程图，虽然看起来仍然复杂，但你不需要完全理解它。你只需知道，这表示我们可以将过程细化，把一个复杂的问题分解成多个简单的小问题，然后按照流程图一步步操作。

## 用 LangGraph + Phi3-vision 实现归回流程

我们可以方便地用 LangGraph 按照上图实现回归分析。以下是 LangGraph 根据添加的节点自动生成的状态图。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/langgraph.jpeg)

我们也可以方便地用 LangChain 的 Nvidia NIM 集成来调用 Phi3-vision。你可以在 [Nvidia NIM](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct) 查看 Phi3-vision 的详细信息，登录后可以在如下位置找到你的 API key。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/nim_key.png)

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
os.environ["NVIDIA_API_KEY"] = ""

llm = ChatNVIDIA(model="microsoft/phi-3-vision-128k-instruct")
```

对于节点的分支路由，我们只需问 Phi3-vision 一个非常简单的问题，比如在 `Constant variance` 节点中，我们会问：“You are a data analysis expert. Does this set of data have constant variance? You only need to answer True or False.”

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/nim_example.png)

在 NIM 的在线测试中，我们可以看到 Phi3-vision 对上面例子的回答是 False。我们只需根据回答的 True 或 False 来决定下一步操作。

在检查数据是否满足正态分布的节点，程序会自动生成 Q-Q plot（一种判断数据是否符合正态分布的图表），然后我们再问 Phi3-vision：“You are a data analysis expert. The attached figure is the Q-Q plot of this set of data. Does this set of data conform to the normal hypothesis? You only need to answer True or False.”

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/nim_q-q_plot.png)

我们得到了 True 的回答，那么根据流程，我们可以知道接下来只需要做加权的回归算法即可。但从图中也可以看出，有多种权重计算方式，我们可以将这些方法列出，并用程序自动生成判断需要的图表，由 Phi3-vision 给我们一个最可能的选项。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/nim_branch.png)

在这个例子中，Phi3-vision 认为第一种权重算法最适合。接下来我们只需自动跳转到对应的算法函数，剩余的工作就是让程序自动计算结果。

这只是一个简单的例子。所有节点都可以有相应的检验算法去验证 Phi3-vision 的回答是否合理。对于需要行业经验的节点，比如权重算法的选择，也可以使用 RAG 提供一些选择的例子。记得 Phi3-vision 可是拥有 128k 的上下文，对于一个简单问题，你可以给它大量的参考例子。

具体的代码实现可以在[这里](../../example/phi3/regress_with_phi3.ipynb) 找到。而下图则是我将最开始的例子用 LangGraph 进行回归的结果。与我自己分析的结果是一致的，但整个过程是自动化的，并且只花费了不到 5 秒的时间。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/resp.png)

## Phi3-vision 的优势

Phi3-vision 相对其他大模型而言非常小，这使得它在多步骤的分析中表现出色，因为单次对话的速度非常快。而且，Phi3-vision 具有 128k 的上下文容量，处理简单问题时可以提供大量参考例子。

对于回归分析这种复杂的项目，没有一个大模型可以一次性给出完整的答案。训练一个可以极少步骤完成项目的大模型也是非常困难的，因为需要收集大量完整的分析过程数据，同时可能还需要描述为何选择某个方法而非其他方法。

回归分析之所以复杂，是因为它需要多次迭代，而每次迭代中遇到的不同问题都有相应的理论和解决方法。但正因为有大量的理论，我们可以方便地将这些问题分解成多个小问题，然后逐一解决。虽然大模型也可以胜任这些问题，但在这种场景下，小模型显然更具优势。

将问题分解后，如果我们需要微调或使用检索增强生成（RAG），我们只需要针对性地收集当前问题的数据，而不必考虑这个问题在整个项目中的位置。

## 结论

通过使用 Phi3-vision 和 LangGraph，我们可以将复杂的回归分析过程分解成多个小问题，并逐步解决。Phi3-vision 在某些节点上提供“主观”的判断，我们再用实现了这些理论的程序进行验证。回归分析本身是一个迭代过程，逐步解决小问题最终能得到完整答案。

这种方法不仅使我们能够随时停下来查看数据情况并进行人工干预，还具有极高的可扩展性。由于 Phi3-vision 处理的是简单问题，我们可以方便地收集足够多的数据进行微调或 RAG，从而在多步骤分析中发挥其优势。总之，Phi3-vision 的快速响应和高上下文容量使其非常适合用于足够复杂但本身却又一套方法论的领域。