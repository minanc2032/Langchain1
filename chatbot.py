#!pip install langchain-core langgraph>0.2.27
#!pip install -qU langchain-openai

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

from langchain_core.messages import HumanMessage

model.invoke([HumanMessage(content="Hi! I'm Bob")])

model.invoke([HumanMessage(content="What's my name?")])

input_message = HumanMessage
input_messages = [HumanMessage(content="Hi I'm Bob"), HumanMessage(content="What's my name?")]
messages = []

for message in input_messages:
     messages.append(message)
     print(messages)
     ai_response = model.invoke(messages)
     print("Human:", message)
     messages.append(ai_response)
     print("AI:", ai_response)

messages =[]

while True:
    user_input = input("You:") 
    if user_input == "exit":
        print("Chat ended")
        break
    messages.append(user_input)

    ai_response = model.invoke(messages)

