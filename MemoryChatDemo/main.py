import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import AzureChatOpenAI

# Initialize LangGraph workflow and memory
workflow = StateGraph(state_schema=MessagesState)
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
)
memory = MemorySaver()

# Define the model-calling function
def call_model(state: MessagesState):
    # Invoke the model
    response = model.invoke(state["messages"])
    return {"messages": response}

# Build and compile the workflow
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=memory)

# Persistent thread ID for memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Chat history in session for UI
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit UI
st.title("LangGraph Chat Bot")

# Display chat history in UI
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_input := st.chat_input("Type your message..."):
    # Show and record the user message
    st.chat_message("user").markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # Build a conversation buffer for the graph from UI history
    from langchain_core.messages import AIMessage
    buffer_msgs = []
    for m in st.session_state.history:
        if m["role"] == "user":
            buffer_msgs.append(HumanMessage(content=m["content"]))
        else:
            buffer_msgs.append(AIMessage(content=m["content"]))

    # Send full history + new human message into LangGraph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    full_response = ""
    for event in app.stream({"messages": buffer_msgs}, config, stream_mode="values"):
        latest = event["messages"][-1]
        if hasattr(latest, "content"):
            full_response = latest.content

    # Display assistant response and record it
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.history.append({"role": "assistant", "content": full_response})

