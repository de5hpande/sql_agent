import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_openai import ChatOpenAI  # Updated to use OpenAI API
from langgraph.graph import StateGraph  # Corrected import
from typing import TypedDict
import pymysql

st.set_page_config(page_title="LangChain + LangGraph: Chat with SQL DB", page_icon="\U0001f9a4")
st.title("\U0001f9a4 LangChain + LangGraph: Chat with SQL DB")

api_key = st.sidebar.text_input(label="OpenAI API Key", type="password")

radio_opt = ["Use SQLite Database - Upload File", "Connect to your MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB you want to chat with", options=radio_opt)

db_uri = "MYSQL" if radio_opt.index(selected_opt) == 1 else "SQLITE"

if db_uri == "MYSQL":
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
    db_file = None
else:
    mysql_host = mysql_user = mysql_password = mysql_db = None
    db_file = st.sidebar.file_uploader("Upload SQLite Database File", type=["db", "sqlite"])

if not api_key:
    st.info("Please add the OpenAI API Key")
    st.stop()

if db_uri == "SQLITE" and not db_file:
    st.info("Please upload a database file to proceed.")
    st.stop()

## LLM model
llm = ChatOpenAI(api_key=api_key, model_name="gpt-4", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, db_file=None, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    try:
        if db_uri == "SQLITE":
            dbfilepath = Path("/tmp/") / db_file.name
            with open(dbfilepath, "wb") as f:
                f.write(db_file.read())
            creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))
        elif db_uri == "MYSQL":
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                st.error("Please provide all MySQL connection details.")
                st.stop()
            return SQLDatabase(create_engine(f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        st.stop()

db = configure_db(db_uri, db_file, mysql_host, mysql_user, mysql_password, mysql_db)

## Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

## Define State Schema
class WorkflowState(TypedDict):
    user_query: str
    sql_response: str
    chatbot_response: str

## Creating LangGraph Workflow
graph = StateGraph(WorkflowState)

def execute_sql(state: WorkflowState):
    """Executes SQL query using the agent"""
    user_query = state["user_query"]
    response = agent.run(user_query)
    return {"sql_response": response}

def chat_response(state: WorkflowState):
    """Generates AI response based on SQL output"""
    sql_response = state["sql_response"]
    
    try:
        return {"chatbot_response": sql_response}
    except Exception as e:
        return {"chatbot_response": f"Error formatting response: {str(e)}"}

graph.add_node("execute_sql", execute_sql)
graph.add_node("chat_response", chat_response)

graph.set_entry_point("execute_sql")
graph.add_edge("execute_sql", "chat_response")

workflow = graph.compile()

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        result = workflow.invoke({"user_query": user_query})
        response = result["chatbot_response"]
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response, unsafe_allow_html=True)

