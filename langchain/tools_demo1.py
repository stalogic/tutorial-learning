"""
参考教程:
https://python.langchain.com/v0.2/docs/how_to/tool_calling/
"""
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.llms import get_llm, LLMS
llm = get_llm(LLMS.GLM4) # 此处可以使用LLMS.GPT4, LLMS.GLM4, LLMS.KIMI


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]


llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12? Also, what is 11 + 49? What is 3 * 5 - 6? Don't do any math yourself, only use tools for math."


messages = [HumanMessage(query)]

while True:
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    if not ai_msg.tool_calls:
        break
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

for i, message in enumerate(messages):
    print(f"Message {i}:\n{[message]}")