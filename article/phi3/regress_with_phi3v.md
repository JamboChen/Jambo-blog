# 用 Phi3-vision 和状态图做简单线性回归

首先，这篇文章并不是回归分析的教程，而是通过回归分析分享我对 Phi3-vision 在应用上的一些想法。虽然会涉及一些回归分析的理论，但这些不是重点，所以不会详细展开。本文的重心在于分享 Phi3-vision 的使用方法，即使不完全理解回归分析的理论也不影响阅读。


## 什么是线性回归

线性回归是一种用于分析和预测数据的方法。简单来说，它试图用一条直线来描述 X 和 Y 之间的关系。

[![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1280px-Linear_regression.svg.png)](https://en.wikipedia.org/wiki/Regression_analysis)

在回归分析的理论中，当数据满足以下条件时，我们可以找到最好的那条直线：
- 残差（Y 与直线的距离）应该服从[常态分布](https://en.wikipedia.org/wiki/Normal_distribution)，即 Y 点应该均匀的分散在直线的两侧，并且不应该离太远，也不应该集中在一侧。
- 残差的方差应该是恒定的，即数据点分散程度应该是一致的。

显然，这两个条件是非常理想的，在现实中很难完全满足。因此，回归分析中除了要寻找潜在的变量，还需要对数据进行变换，使其尽可能满足这两个条件。

### 例子

下面是一个简单的例子来展示数据的分布：

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/example.png)

从图中，我们可以很容易得出两个结论：
1. 随着 X 越大，Y 也呈现增大的趋势。
2. 左边的点比右边的点更集中。意味着当 X 越大时，Y 的变化范围也越大。

这张图并不满足上面提到的条件，因此如果直接用线性回归方法去拟合这组数据，结果可能不理想。下图就是不经过调整直接计算回归线得到的结果。请关注 `R-squared` 值，这个值越接近 1，说明模型越好。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/lse.png)

通过一些变换方法，我们可以得到一个更好的模型。回归分析中有很多理论和检验方法来判断数据应该使用哪些变换方法，但在真实应用中，仍然需要一些主观判断和经验。


## 为什么我认为 Phi3-vision 适合做回归分析

虽然回归分析的理论非常严谨，但在实际应用中，往往需要主观判断。现实中的数据并不如理论那般完美，充满不可预知的变数。因此，回归分析并不是在找数据间的因果关系，而是提供一种最可靠的方式去“猜测”数据。

正如“所有模型都是错误的”，我们只是在寻找一个适合我们数据的模型，这过程中需要很多主观判断和行业经验。Phi3-vision 能够快速地根据图表给我们一些“主观”的判断，这在实际应用中非常有帮助。当我们有了“主观”判断后，我们也可以用回归理论中的一些检验方式来验证这个判断是否合理。

## 回归分析的流程

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/flow.png)

这是一个简化的流程图，虽然看起来仍然复杂，但你不需要完全理解它。你只需知道，这表示我们可以将过程细化，把一个复杂的问题分解成多个简单的小问题，然后按照流程图一步步操作。

## 用 LangGraph + Phi3-vision 实现基于状态图的回归流程

我们可以方便地用 LangGraph 按照上图实现回归分析。以下是 LangGraph 根据添加的节点自动生成的状态图。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/langgraph.jpeg)

如果你本地没有 Phi3-vision 的运行环境，我们也可以方便地用 LangChain 的 Nvidia NIM 集成来调用模型，这样可以快速检验特定模型是否适合这项应用。你可以在 [Nvidia NIM](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct) 查看 Phi3-vision 的详细信息，登录后可以在如下位置找到你的 API key。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/nim_key.png)

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
os.environ["NVIDIA_API_KEY"] = ""

llm = ChatNVIDIA(model="microsoft/phi-3-vision-128k-instruct")
```

对于节点的分支路由，我们只需问 Phi3-vision 一个非常简单的问题，比如在 `Constant variance` 节点中，我们会问：“You are a data analysis expert. Does this set of data have constant variance? You only need to answer True or False.”

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/nim_example.png)

在 NIM 的在线测试中，我们可以看到 Phi3-vision 对上面例子的回答是 False。我们只需根据回答的 True 或 False 来决定下一步操作。

在检查数据是否满足正态分布的节点，我们会编写程序来自动更具数据生成 Q-Q plot（一种判断数据是否符合正态分布的图表），然后我们再问 Phi3-vision：“You are a data analysis expert. The attached figure is the Q-Q plot of this set of data. Does this set of data conform to the normal hypothesis? You only need to answer True or False.”

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/nim_q-q_plot.png)

我们得到了 True 的回答，那么根据流程，我们可以知道接下来只需要做加权的回归算法即可。但从图中也可以看出，有多种权重计算方式，我们可以将这些方法列出，并用程序自动生成判断需要的图表，由 Phi3-vision 给我们一个最可能的选项。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/nim_branch.png)

在这个例子中，Phi3-vision 认为第一种权重算法最适合。接下来我们只需自动跳转到对应的算法函数，剩余的工作就是让程序自动计算结果。

具体的代码实现可以在[这里](../../example/phi3/regress_with_phi3.ipynb)找到。而下图则是我将最开始的例子用 LangGraph 进行回归的结果。可以看到 `R-squared` 值从 0.408 提升到 0.521，而整个过程是用了不到 5s 的时间，如果是在本地部署的模型上推理，时间可能会更短。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/regress_with_phi3v/resp.png)

## 例子中的重点

### 分解问题

对于回归分析这种复杂的项目，没有一个大模型可以一次性给出完整的答案。训练一个可以极少步骤完成项目的大模型也是非常困难的，因为需要收集大量完整的分析过程数据，同时可能还需要描述为何选择某个方法而非其他方法。

我们能自动化的快速解决这样的问题，很大程度上是因为我们将这个问题分解的足够细致。每个节点都是非常简单的问题，Phi3-vision 只需要回答 True 或 False。这样我们可以方便地验证 Phi3-vision 的回答是否合理。

同时因为我们将问题分解的足够细致，我们可以针对每个问题编写针对性的提示文本或 RAG，可以更好地引导 Phi3-vision 的回答。

### Phi3-vision 的优势

由于问题被分解得非常细致，每个问题都相对简单。在这种场景下，擅长回答复杂问题的大模型不如快速的小模型实用。而且，Phi3-vision 具有 128k 的上下文容量，处理简单问题时可以提供大量参考例子。因为问题都是已知的，我们可以将例子硬编码，而不必等待嵌入模型和向量数据库来匹配适合的例子。

### 状态图的优势

回归分析之所以复杂，不仅因为我们要寻找不同因素之间的关联，还需要不断优化模型、转换数据，并针对修改后的结果继续优化。很多简单的场景也非常依赖循环，比如我们可能需要不断检查是否获得了足够的信息来进行下一步操作，或者不断调整模型的参数。这些都是状态图的优势。

目前大部分主流的基于 LLM 的可视化工具都是基于 pipeline，比如 prompt flow 和 langflow。在需要循环的地方则依靠 Agent 或代码实现。这在某种程度上限制了我们的操作，甚至会将一些问题复杂化。状态图和 pipeline 并不是互斥的，状态图可以作为 pipeline 更高层的抽象，从而将处理不同任务的 pipeline 连接起来。

## 与 Agent 的区别

Agent 让模型通过总结当前获得的信息来自主决定下一步操作，赋予了模型自主决策和主动收集信息的能力。此外，也有通过多代理方法解决复杂问题的方案，但这种方法更适合用于没有明确解决方案时探索方法。对于本身有一套方法论的领域，使用 Agent 可能显得多余，并且会增加系统的复杂度。

由于每个步骤都是由模型自己主导，为了做出正确的决策，模型需要生成大量的思考过程文字，这会导致系统运行时间变长、消耗大量的 tokens，并且仍然存在跑偏的问题。尽管可以通过调整提示来影响模型的决策，但调试 prompt 比训练模型更为黑箱，你很难知道提示究竟影响了模型的哪个部分。

本文中的方法由工程师提前设计框架流程，模型只在分支选择上发挥作用，并且可以通过自动化验证保证模型的选择是正确的。整套系统基于一张状态图，因此我们可以结构化地记录整个运行过程，而不仅仅是记录 LLM 输出的文字，以便后续调试和验证。由于模型遇到的问题是可控的，且输出仅是布尔值或分支选择，我们可以通过示例来调整模型的选择。即使模型给出了错误的结果，运行流程也在可预期的范围内，因此我们可以很容易地找到问题所在。

## 结论

通过使用 Phi3-vision 和 LangGraph，我们可以将复杂的回归分析过程分解成多个小问题，并逐步解决。Phi3-vision 在某些节点上提供“主观”的判断，我们再用实现了这些理论的程序进行验证。回归分析本身是一个迭代过程，逐步解决小问题最终能得到完整答案。

这种方法不仅使我们能够随时停下来查看数据情况并进行人工干预，还具有极高的可扩展性。由于 Phi3-vision 处理的是简单问题，我们可以方便地收集足够多的数据进行微调或 RAG，从而在多步骤分析中发挥其优势。总之，Phi3-vision 的快速响应和高上下文容量使其非常适合用于足够复杂但本身具有一套方法论的领域。

当然，这种方法并不适合在未知领域探索解决方法，因为我们需要提前设计好整个流程。但对于遇到的问题有已知方案的领域，用 Agent 可能会显得有些多余，而用 Phi3-vision 和状态图则会更加高效。但这也并不代表这两种方案是互斥的，我们可以根据具体情况选择合适的方法。

以上这只是我的一个想法，希望能给大家一些启发。如今还没有诞生基于图的可视化工具可能是因为一些我没有考虑到的问题，或者是我对现有工具的了解不够。如果你有其他想法或者建议，欢迎在评论区留言。