from langchain_groq import ChatGroq
from langchain.memory import SQLChatMessageHistory

memory = SQLChatMessageHistory(
    session_id="1234",
    table_name="chat_history",
    connection="sqlite:///chat_history.db",)
agent = ChatGroq(model="llama-3.3-70b-versatile")

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    memory.add_user_message(user_input)
    ai_input =memory.get_messages()[-5:]
    # print (ai_input)
    response = agent.invoke(ai_input)
    print(" - "*60)
    memory.add_ai_message(response.content)
    print("Agent:", response.content)

for message in memory.messages:
    message.pretty_print()