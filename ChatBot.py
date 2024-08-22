import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
os.environ["OPENAI_API_KEY"] = getpass.getpass()



from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough



store = {} #dictionary that holds chat histories for corresponding session IDs

config = {"configurable": {"session_id": "abc2"}} #sets the sessionID

#this function checks if we have that session_id in our store, and if so, we will recover the messages from that session_id's history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


model = ChatOpenAI(model="gpt-3.5-turbo")

#tokens are number of words essentially. Once we pass 65, we get rid of the older ones and use the newer ones.
trimmer = trim_messages(
    max_tokens=500,
    strategy="last", #what allows us to keep the newest
    token_counter=model, #it's counting tokens from the model we selected
    include_system=True, #includes system messages
    allow_partial=False, #partial tokens should get deleted
    start_on="human", #it won't delete the first system messages, it will only consider from the first human message down.
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)




#prompt the system itself
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        #this is what we pass all the messages into
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)




with_message_history = RunnableWithMessageHistory(chain, 
                                                  get_session_history, 
                                                  input_messages_key="messages",)


response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="what math problem did I ask?")],
        "language": "English",
    },
    config=config
)

print(response.content)