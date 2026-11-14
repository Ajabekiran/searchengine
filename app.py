import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun 
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os 
from dotenv import load_dotenv

##
##arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

##sidebar for setting 
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("enter your groq API key", type="password")

if "messages" not in st.session_state :
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi I am a chatbot who can search on web. How may I help you today"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="what is machine learning"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = api_key, model ="openai/gpt-oss-120b",streaming=True)
    tools = [search, wiki, arxiv]

    # Tool calling agents require a ChatPromptTemplate with an agent_scratchpad
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can search the web, Wikipedia, and Arxiv."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)

    search_agent = AgentExecutor(agent=agent, tools=tools, verbose = True)

    with st.chat_message("assistant"):
        st_cb= StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        chat_history = [
            (msg["role"], msg["content"]) 
            for msg in st.session_state.messages[:-1]
        ]
        response = search_agent.invoke(
            {"input": prompt, "chat_history": chat_history}, 
            config = {"callbacks":[st_cb]}
        )

        # Extract the final string output from the response dictionary
        output_content = response["output"]
        st.session_state.messages.append({"role":"assistant","content":output_content})
        st.write(output_content)
