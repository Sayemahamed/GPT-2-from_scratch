from groq import Groq
from groq.types.chat.chat_completion import ChatCompletion
import rich
from xml.etree import ElementTree as ET
import re
client = Groq()

# chat_completion: ChatCompletion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of low latency LLMs",
#         }
#     ],
#     model="llama-3.3-70b-versatile",
# )
# rich.print(chat_completion)
# rich.print(chat_completion.choices[0].message.tool_calls)
def calculate(expression: str) :
    return eval(expression)

# calculate("(2+2)/4")

def weather(location: str) :
    return f"The weather in {location} is sunny"

messages = [
    {
        "role": "system",
        "content": """You are a helpful and intelligent agent designed to assist users efficiently. You have access to the following tools to provide accurate and useful information:

Mathematical Evaluations: Use the calculate tool to evaluate mathematical expressions.
Example call:
<action>
  <tool> calculate </tool> <parameter> (2+2)/4 </parameter>
</action>
Weather Information: Use the weather tool to fetch real-time weather details.
Example call:
<action> 
     <tool> weather </tool> <parameter>  New York </parameter> 
</action>
Your goal is to respond accurately and concisely while maintaining a friendly and helpful tone.""",
    },
    {
        "role": "user",
        "content": "Solve the expression: (4*8)/(2+2)",
    },
    {
        "role": "user",
        "content": "What is the weather in New York?",
    },
]

chat_completion: ChatCompletion = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile",
)
text = chat_completion.choices[0].message.content
def extract_actions(text):
    pattern = r"<action>[\s\S]*?</action>"  
    return re.findall(pattern, text) 

actions = extract_actions(text)
# rich.print(chat_completion)

def parse_action(action_str):
    try:
        root = ET.fromstring(action_str)
        tool = root.find("tool").text.strip()
        parameter = root.find("parameter").text.strip()
        return {"tool": tool, "parameter": parameter}
    except Exception as e:
        return {"error": f"Failed to parse action: {e}"}


for action in actions:
    parsed_action = parse_action(action)
    if parsed_action["tool"] == "calculate":
        expression: str = parsed_action["parameter"]
        result = calculate(expression)
        print(f"Result: {result}")
    elif parsed_action["tool"] == "weather":
        location: str = parsed_action["parameter"]
        result: str = weather(location)
        print(f"Result: {result}")
