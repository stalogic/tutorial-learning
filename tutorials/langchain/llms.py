from langchain_openai import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from enum import Enum

OLLAMA_HOST = "http://172.19.21.52:11434"

class LLMS(Enum):
    QWEN = "qwen:7b"
    LLAMA3 = "llama3:8b"
    PHI3 = "phi3:3.8b"

    GPT4 = "gpt-4"
    GLM4 = "glm-4"
    KIMI = "moonshot-v1-8k"


def get_llm(model: str | LLMS):
    match model:
        case LLMS.QWEN:
            llm = ChatOllama(model=LLMS.QWEN.value, base_url=OLLAMA_HOST)
        case LLMS.LLAMA3:
            llm = ChatOllama(model=LLMS.LLAMA3.value, base_url=OLLAMA_HOST)
        case LLMS.PHI3:
            llm = ChatOllama(model=LLMS.PHI3.value, base_url=OLLAMA_HOST)
        case LLMS.GPT4:
            llm = ChatOpenAI(model_name=LLMS.GPT4.value, openai_api_key="sk-RPLgjF1NQbKh7HWB37Af4957A58546CdBf59E96f6f60F010", base_url="https://openai.sohoyo.io/v1")
        case LLMS.GLM4:
            llm = ChatOpenAI(model_name=LLMS.GLM4.value, openai_api_key="2cc262ec16952966f69d6c5330af0572.GVZ6fWvlEV64Km69", base_url="https://open.bigmodel.cn/api/paas/v4/")
        case LLMS.KIMI:
            llm = ChatOpenAI(model_name=LLMS.KIMI.value, openai_api_key="sk-gw5WVfZPAgAGJuAosCEkT1YWwbyhb3Ui4iwCOUWwrp8XxL90", base_url="https://api.moonshot.cn/v1")
    return llm


if __name__ == "__main__":
    for model in LLMS:
        print(model, type(model), model.value)
        llm = get_llm(model)
        print(llm.invoke("使用中文介绍一下你自己").content)