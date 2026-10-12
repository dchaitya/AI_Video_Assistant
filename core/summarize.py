from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

def get_llm(model: str = 'qwen/qwen3.8-27b', temperature: float = 0.3):
    return ChatGroq(model=model, temperature=temperature)

def split_transcript(transcript:str)->list:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)

def summarize(transcript: str, model: str = 'qwen/qwen3.8-27b', temperature: float = 0.3) -> str:
    llm = get_llm(model=model, temperature=temperature)
    map_prompt=ChatPromptTemplate.from_messages(
        [
            ("system","Summarize this portion of text"),
            ("human","{text}"),
            
        ]
    )
    map_chain=map_prompt | llm | StrOutputParser()

    chunks=split_transcript(transcript)

    chunk_summaries=[map_chain.invoke({"text":chunk}) for chunk in chunks]

    combined="\n".join(chunk_summaries)

    combined_prompt=ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are an expert meeting summarizer.Combine these Partial summaries 
            into one final professional meeting summary in bullet points.
            """),
            ("human","{text}"),
        ]
    )

    combined_chain=(
        RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )
    return combined_chain.invoke(combined)

def generate_title(transcript: str, model: str = 'qwen/qwen3.8-27b') -> str:
    llm = get_llm(model=model, temperature=0.2)
    title_prompt=ChatPromptTemplate.from_messages([
        ("system",
         "Based on the meeting transcript, generate a short proffesional title (max 8 words).only return title nothing else."),
         ("human","{text}"),
    ])
    title_chain=(
        RunnableLambda(lambda x:{"text":x})| title_prompt| llm| StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])