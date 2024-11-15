
> [!WARNING]
> Project is in its early stage of development. Current version is not stable.

# 🐈‍⬛ nyai
Versative AI client and toolkit that works out of the box. Work with AI models from different providers seamlessly. Powered by OpenAI's Python SDK behind the scenes to handle dozens of AI providers without all the complexity. 

Supports the following out of the box:
- OpenAI
- SambaNova
- Anthropic
- Cohere
- Fireworks
- Together
- Vertex AI

## 📦 Setup
```shell
pip install nyai
```

## 🛠️ Usage
Set up a base client from one of the supported providers.
```python
from nyai import Client

client = Client(provider="sambanova",
                api_key="API_KEY")
```
Extend these clients with better functionality.
```python
from nyai.llm import LLM

llm = LLM(client, model="Meta-Llama-3.2-3B-Instruct")
for chunk in llm.stream("Tell me a joke"):
    print(chunk, end="")

```
Want to add your own Provider?
```python
from nyai.providers import Provider
from nyai import Client

perplexity = Provider(name="Perplexity", endpoint="https://api.perplexity.ai")
client = Client(provider=perplexity, api_key="API_KEY")
```

*Docs to be continued...*

## 🎯 Motivation
Handling LLMs or other AI models from different providers *should* be easy. `Nyai` aims to be a dirt simple solution and toolkit that makes interactions with multiple AI providers seamless. Why not use `LangChain` instead? *Why would you do that to yourself.* `Nyai` was also originally built for `Openmacro`.