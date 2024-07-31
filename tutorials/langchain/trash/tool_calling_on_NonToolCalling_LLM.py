from langchain_community.llms import Ollama
from langchain.llms import LLMS, get_llm

# model = Ollama(model="qwen:0.5b", base_url = "http://localhost:11434")
model = get_llm(LLMS.GLM4)

from langchain_core.tools import tool


@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers together."""
    return x * y


@tool
def add(x: int, y: int) -> int:
    "Add two numbers."
    return x + y


tools = [multiply, add]

# Let's inspect the tools
for t in tools:
    print("--")
    print(t.name)
    print(t.description)
    print(t.args)

print(multiply.invoke({"x": 4, "y": 5}))


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description

rendered_tools = render_text_description(tools)
print(rendered_tools)

system_prompt = f"""\
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

The `arguments` should be a dictionary, with keys corresponding 
to the argument names and the values corresponding to the requested values.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)


chain = prompt | model
message = chain.invoke({"input": "what's 3 * 1132"})

# Let's take a look at the output from the model
# if the model is an LLM (not a chat model), the output will be a string.
if isinstance(message, str):
    print(message)
else:  # Otherwise it's a chat model
    print(message.content)


from langchain_core.output_parsers import JsonOutputParser

chain = prompt | model | JsonOutputParser()
resp = chain.invoke({"input": "what's 3 * 1132"})
print(resp)


from typing import Any, Dict, Optional, TypedDict

from langchain_core.runnables import RunnableConfig


class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: Dict[str, Any]


def invoke_tool(
    tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None
):
    """A function that we can use the perform a tool invocation.

    Args:
        tool_call_request: a dict that contains the keys name and arguments.
            The name must match the name of a tool that exists.
            The arguments are the arguments to that tool.
        config: This is configuration information that LangChain uses that contains
            things like callbacks, metadata, etc.See LCEL documentation about RunnableConfig.

    Returns:
        output from the requested tool
    """
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)


# res = invoke_tool({"name": "multiply", "arguments": {"x": 3, "y": 5}})
# print(res)


# chain = prompt | model | JsonOutputParser() | invoke_tool
# res = chain.invoke({"input": "what's thirteen times 4.14137281"})
# print(res)

from langchain_core.runnables import RunnablePassthrough

chain = (
    prompt | model | JsonOutputParser() | RunnablePassthrough.assign(output=invoke_tool)
)
res = chain.invoke({"input": "what's thirteen times 4.14137281"})
print(res)
