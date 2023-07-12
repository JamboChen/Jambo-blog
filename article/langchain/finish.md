上一篇教程我们介绍了 ReAct 系统，这是一个非常强大的行为模式，但它需要编写大量的示例来告诉 LLM 如何思考、行动，并且为了遵循这个模式，还需要编写代码来分析生成文字、调用函数、拼接 prompt 等，这些工作都是十分繁琐且冗长的。而 LangChain 帮我们把这些步骤都隐藏起来了，将这一系列动作都封装成 “代理”，我们只需要提供有那些工具可以使用，以及这些工具的功能，就可以让 LLM 自动完成这些工作了。这样我们的代码就会变得简洁、易读，并且拥有更高的灵活性。

# 代理（Agent）

所谓代理，就是将一系列的行为和可以使用的工具封装起来，让我们可以通过一个简单的接口来调用这些动作，而不需要关心这些动作是如何完成的。这样可以让我们的代码更简洁、易读，并且拥有更高的灵活性。接下来我们就结合 ReAct 和向量数据库来介绍代理的使用。
