import streamlit as st
import pymysql
pymysql.install_as_MySQLdb()  
import re
import warnings
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Database Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Database Assistant")

# Retrieve API Key and Database URI from Streamlit Secrets
gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
db_uri = st.secrets.get("MYSQL_URI", "")

if not gemini_api_key or not db_uri:
    st.error("Missing Secrets: Please configure GEMINI_API_KEY and MYSQL_URI.")
    st.stop()

# Role-Based Access Control (RBAC) Setup
role = st.sidebar.selectbox("Role:", ["Sales_Rep", "Inventory_Manager"])
allowed_tables = ["customers", "orders", "products"] if role == "Sales_Rep" else ["products"]
st.sidebar.info(f"Permitted Tables: {', '.join(allowed_tables)}")

def get_db():
    return SQLDatabase.from_uri(
        db_uri,
        include_tables=allowed_tables,
        sample_rows_in_table_info=2,
        engine_args={
            "connect_args": {
                "ssl": {"ca": None} 
            }
        }
    )

def clean_sql(raw: str) -> str:
    cleaned = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
    return cleaned.replace("```", "").strip()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about the database (English / Sinhala)...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("AI analyzing database..."):
            try:
                db = get_db()
                
                # Initialize Gemini LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.5-flash-lite", 
                    google_api_key=gemini_api_key, 
                    timeout=30.0
                )

                # 1. SQL Generation with RBAC and Scope Rules
                sql_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an Enterprise MySQL AI Assistant.
The current logged-in role is '{current_role}'.
This role has permission ONLY for these tables: {permitted_tables}.
The full enterprise database contains: 'customers', 'orders', and 'products'.

STRICT RULES:
1. If the user's question asks for data in tables outside their permitted scope (for example, an Inventory Manager asking about customers or orders/sales):
   Output EXACTLY: PERMISSION_DENIED
2. If the question is completely unrelated to the company database (e.g. general knowledge, politics, personal questions):
   Output EXACTLY: NOT_RELEVANT
3. If the question is valid and within their permitted schema ({schema}):
   Output ONLY the raw executable MySQL query. No markdown, no commentary, no destructive statements."""),
                    ("human", "{question}")
                ])
                sql_chain = sql_prompt | llm | StrOutputParser()
                generated_sql = clean_sql(sql_chain.invoke({
                    "current_role": role,
                    "permitted_tables": ", ".join(allowed_tables),
                    "schema": db.get_table_info(), 
                    "question": user_query
                }))

                # Handle Permission Denied (RBAC Restriction)
                if "PERMISSION_DENIED" in generated_sql.upper():
                    rbac_prompt = ChatPromptTemplate.from_messages([
                        ("system", """Inform the user that access is denied based on their role.
State clearly:
- Their current role is '{current_role}'.
- They only have access to: {permitted_tables}.
- They do not have permission to view other data (like customers or orders).

STRICT LANGUAGE RULE:
- If the user asked in English, reply in English.
- If the user asked in Sinhala, reply in natural Sinhala."""),
                        ("human", "{question}")
                    ])
                    rbac_chain = rbac_prompt | llm | StrOutputParser()
                    response = rbac_chain.invoke({
                        "current_role": role,
                        "permitted_tables": ", ".join(allowed_tables),
                        "question": user_query
                    })

                    st.error(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                # Handle General Knowledge / Out-of-Scope Queries
                elif "NOT_RELEVANT" in generated_sql.upper():
                    scope_prompt = ChatPromptTemplate.from_messages([
                        ("system", """Politely explain to the user that you can only answer questions related to the company database.
STRICT LANGUAGE RULE:
- If the user asked in English, reply in English.
- If the user asked in Sinhala, reply in natural Sinhala."""),
                        ("human", "{question}")
                    ])
                    scope_chain = scope_prompt | llm | StrOutputParser()
                    response = scope_chain.invoke({"question": user_query})

                    st.warning(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                # Handle Valid Allowed Queries
                else:
                    with st.expander("🛠️ Generated SQL"):
                        st.code(generated_sql, language="sql")

                    try:
                        db_result = db.run(generated_sql)
                    except Exception as db_err:
                        db_result = f"Query Execution Error: {str(db_err)}"

                    ans_prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a professional enterprise database assistant.
Synthesize a concise, accurate answer based on the SQL query and its execution result.

CRITICAL LANGUAGE RULE:
- Always reply in the EXACT language used in the user's question.
- If the question is in English, reply entirely in English.
- If the question is in Sinhala, reply in natural Sinhala."""),
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