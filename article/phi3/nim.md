# 使用英伟达 NIM 与微软 Phi-3-vision 进行 OCR 识别

Phi-3-Vision-128K-Instruct 是一种轻量级、最先进的开放式多模态模型，它基于包括合成数据和经过筛选的公开网站在内的数据集构建，重点关注文本和视觉方面的高质量、推理密集型数据。该模型属于 Phi-3 模型系列，多模态版本可支持 128K 上下文长度（以 token 为单位）。该模型经过了严格的增强过程，结合了监督微调和直接偏好优化，以确保精确遵循指令和采取强大的安全措施。

资源和技术文档：

-  [Phi-3 微软博客](https://azure.microsoft.com/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/?WT.mc_id=studentamb_228125)
-  [Phi-3 技术报告](https://aka.ms/phi3-tech-report)
-  [Azure AI Studio 上的 Phi-3](https://aka.ms/try-phi3vision)
-  [Phi-3 食谱](https://github.com/microsoft/Phi-3CookBook)

 
## NVIDIA NIM API

NVIDIA NIM API 是一种用于构建和部署自定义 AI 模型的 API,它旨在简化模型训练和部署的复杂性，使开发人员能够专注于模型的设计和性能优化。NIM 提供了一种简单的方式来训练和部署模型，以便在边缘设备上进行推理。比如本教程的撰写，是在 Nvidia Jetson NX 上进行的。

1. 前往 [NVIDIA NIM Phi-3-vision](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct) 页面。
2. 页面右上角点击 "Login" 登录。
3. 点击 "Python" 选项卡，并点击 "Get API Key" 按钮。![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/ocr/1.png)
4. 点击 "Generate Key"，复制并保存你的 API Key。![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/img/phi3/ocr/2.png)

## ORC 识别

我现在手上有一张图片，上面是某食品包装的营养成分表。

![alt text](https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/example/phi3/nutrition_facts.jpg)

我希望让 Phi-3-vision 模型识别这张图片，并将图片上的表格转换成 Markdown 格式的表格。

```python
import base64 # 用于编码图片
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TOKEN") # 从环境变量中获取 API Key
```

复制 NIM 页面中的 Python 代码，并将其包装成函数

```python
def invoke(prompt: str, image_b64: str, stream=True):
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct"

    assert (
        len(image_b64) < 180_000
    ), "To upload larger images, use the assets API (see docs)"

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "text/event-stream" if stream else "application/json",
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'{prompt} <img src="data:image/png;base64,{image_b64}" />',
            }
        ],
        "max_tokens": 512,
        "temperature": 1.00,
        "top_p": 0.70,
        "stream": stream,
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    if stream:
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8')[6:])
                print(data["choices"][0]["delta"]["content"])
    else:
        print(response.json()["choices"][0]["message"]["content"])
```

我们需要将图片读进 Python，并转换成 base64 编码

```python
with open("nutrition_facts.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
```

调用 `invoke` 函数

```python
invoke("Help me organize the table in the picture into md format", image_b64, stream=False)
```

运行得到以下输出：

> Certainly! Below is the markdown table converted from the image provided:
> 
> ```markdown
> | Nutrition Facts                |                     |
> |--------------------------------|---------------------|
> | 8 servings per container       |                     |
> | Serving size                   | 2 rolls (20g)       |
> | **Amount Per Serving**         |                     |
> | Calories                       | 120                 |
> | **% Daily Value***             |                     |
> | Total Fat                      | 9g                  |
> |                                | 12%                 |
> | Saturated Fat                  | 4g                  |
> |                                | 20%                 |
> | Trans Fat                      | 0g                  |
> | Cholesterol                     | 0mg                 |
> |                                | 0%                  |
> | Sodium                         | 55mg                |
> |                                |
> | Trans Fat                      | 0g                  |
> | Cholesterol                    | 0mg                 |
> |                                | 0%                  |
> | Sodium                         | 55mg                |
> |                                | 2%                  |
> | Total Carbohydrate             | 10g                 |
> |                                | 4%                  |
> | Dietary Fiber                  | 0g                  |
> |                                | 0%                  |
> | Total Sugars                   | 1g                  |
> | Includes 1g Added Sugars       |                     |
> |                                | 2%                  |
> | Protein                        | less than 1g        |
> | Vit. D                         | 0mcg 0%             |
> | Calcium                        | 0mg 0%              |
> | Iron                           | 0mg 0%              |
> | Potas.                         | 0mg 0%              |
> ```
> 
> *The % Daily Value tells you how much a nutrient in a serving of food > contributes to a daily diet. 2, 2,0 calories a day is used for general > nutration advice


使用英伟达 NIM 与微软 Phi-3-vision 进行OCR识别的应用场景非常广泛，它们可以极大地提高自动化处理和数据提取的效率。以下是一些具体的应用场景：
- 文档自动化处理：在办公环境中，大量纸质文档需要被转换为数字格式以便进行存储、搜索和共享。使用NVIDIA NIM与Phi-3-vision进行OCR识别，可以快速准确地将纸质文档（如合同、发票、报告等）转换为可编辑的文本或Markdown格式，提高办公效率。
- 产品标签识别：在零售和物流领域，产品标签上通常包含重要的产品信息，如名称、生产日期、保质期、条形码等。通过OCR技术，这些标签信息可以被快速准确地提取出来，用于库存管理、物流跟踪和产品追溯。
- 财务数据处理：在处理财务报表、发票和收据等财务文件时，OCR技术可以自动识别和提取其中的关键信息，如金额、日期、客户名称等。这有助于加快数据处理速度，减少人为错误，提高财务工作效率。
- 图书资料数字化：图书馆和档案馆中存储着大量的纸质书籍和资料，这些资料需要被数字化以便进行在线访问和保存。使用OCR技术可以自动识别和转换书籍和资料中的文本内容，实现图书资料的快速数字化。
- 表单处理：在企业或政府部门中，经常需要处理各种表单（如申请表、调查问卷等）。使用OCR技术可以自动识别表单中的文本和图像内容，将其转换为可编辑的数据格式，方便后续的数据分析和处理。
- 自动化安全检查：在安全检查领域，如机场、车站等场所，需要识别旅客的身份证件、车票等信息。通过OCR技术可以自动识别和提取这些信息，提高安全检查的速度和准确性。
- 无障碍阅读辅助：对于视力障碍人群，OCR技术可以将纸质书籍、杂志等转换为可听的语音格式，帮助他们更方便地获取信息。同时，OCR技术还可以用于制作电子书和网页的无障碍版本，提高网站和应用的可用性。

这些只是使用英伟达 NIM 与微软 Phi-3-vision 进行OCR识别的一些典型应用场景。随着技术的不断发展和应用场景的不断拓展，OCR技术将在更多领域发挥重要作用。