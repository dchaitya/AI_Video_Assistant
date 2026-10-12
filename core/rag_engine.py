import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from core.vector_store import build_vector_store,load_vector_store,get_retriever
from dotenv import load_dotenv
load_dotenv()

def get_llm(model: str = "qwen/qwen3.8-27b", temperature: float = 0.3):
    return ChatGroq(
        model=model,
        temperature=temperature
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(
    transcript: str = None,
    vector_store = None,
    retriever = None,
    collection_name: str = "meeting_transcript",
    k: int = 4,
    model: str = "qwen/qwen3.8-27b",
    temperature: float = 0.3,
):
    if retriever is None:
        if vector_store is None:
            if not transcript:
                raise ValueError("Either transcript, vector_store, or retriever must be provided.")
            vector_store = build_vector_store(transcript, collection_name=collection_name, clear_existing=False)
        retriever = get_retriever(vector_store, k=k)

    llm = get_llm(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
]
    )

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
         | prompt | llm | StrOutputParser()
    )

    return rag_chain

def load_chain(collection_name: str = "meeting_transcript", k: int = 4, model: str = "qwen/qwen3.8-27b", temperature: float = 0.3):
    vector_store = load_vector_store(collection_name=collection_name)
    retriever = get_retriever(vector_store, k=k)
    llm = get_llm(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
]
    )

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
        }
        | prompt | llm | StrOutputParser()
    )
    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    answer = rag_chain.invoke(question)
    return answer

def ask_question_with_sources(rag_chain, retriever, question: str, collection_name: str = None):
    docs = []
    if retriever is not None:
        try:
            docs = retriever.invoke(question)
        except Exception:
            if collection_name:
                try:
                    vs = load_vector_store(collection_name)
                    docs = get_retriever(vs).invoke(question)
                except Exception:
                    docs = []
    answer = ask_question(rag_chain, question)
    return answer, docs

