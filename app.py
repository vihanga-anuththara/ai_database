import streamlit as st
import re
import warnings
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Database Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Database Assistant")

# Get API Key and URI from Secrets
gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
db_uri = st.secrets.get("MYSQL_URI", "")

if not gemini_api_key or not db_uri:
    st.error("Missing Secrets: Please configure GEMINI_API_KEY and MYSQL_URI.")
    st.stop()

# RBAC
role = st.sidebar.selectbox("Role:", ["Sales_Rep", "Inventory_Manager"])
allowed_tables = ["customers", "orders", "products"] if role == "Sales_Rep" else ["products"]
st.sidebar.info(f"Allowed: {', '.join(allowed_tables)}")

db = SQLDatabase.from_uri(db_uri, include_tables=allowed_tables, sample_rows_in_table_info=2)

def clean_sql(raw):
    cleaned = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
    return cleaned.replace("```", "").strip()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Database එකෙන් දැනගන්න ඕන දේ අහන්න (සිංහල / English)...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("AI analyzing database..."):
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=gemini_api_key, timeout=30.0)

                # SQL Prompt
                sql_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a MySQL expert. Output ONLY valid raw SQL query to answer the question.\n\nSchema:\n{schema}"),
                    ("human", "{question}")
                ])
                sql_chain = sql_prompt | llm | StrOutputParser()
                generated_sql = clean_sql(sql_chain.invoke({"schema": db.get_table_info(), "question": user_query}))

                with st.expander("🛠️ Generated SQL"):
                    st.code(generated_sql, language="sql")

                db_result = db.run(generated_sql)

                # Answer Prompt
                ans_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer concisely based on the query and result. If queried in Sinhala, reply in natural Sinhala."),
                    ("human", "Question: {question}\nSQL: {query}\nResult: {result}")
                ])
                ans_chain = ans_prompt | llm | StrOutputParser()
                response = ans_chain.invoke({"question": user_query, "query": generated_sql, "result": db_result})

                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err})
