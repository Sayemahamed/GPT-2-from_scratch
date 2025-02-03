from groq import Groq
from groq.types.chat.chat_completion import ChatCompletion

client = Groq()

chat_completion: ChatCompletion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model="llama-3.3-70b-versatile",
)
print(chat_completion.choices[0].message.content)