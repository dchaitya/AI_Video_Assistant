from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
load_dotenv()

def get_llm(model: str = "qwen/qwen3.8-27b", temperature: float = 0.3):
    return ChatGroq(model=model, temperature=temperature)

def build_chain(system_prompt: str, model: str = "qwen/qwen3.8-27b", temperature: float = 0.3):
    llm = get_llm(model=model, temperature=temperature)
    return (
        RunnableLambda(lambda x: {"text": x}) | 
        ChatPromptTemplate([
            ("system", system_prompt),
            ("human", "{text}")
        ])
        | llm | StrOutputParser()
    )

action_prompt= ("You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'")

question_prompt=("From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
        )

decision_prompt=(
    "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
)
def extract_action_items(transcript: str, model: str = "qwen/qwen3.8-27b", temperature: float = 0.3) -> str:
    chain = build_chain(action_prompt, model=model, temperature=temperature)
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str, model: str = "qwen/qwen3.8-27b", temperature: float = 0.3) -> str:
    chain = build_chain(decision_prompt, model=model, temperature=temperature)
    return chain.invoke(transcript)

def extract_questions(transcript: str, model: str = "qwen/qwen3.8-27b", temperature: float = 0.3) -> str:
    chain = build_chain(question_prompt, model=model, temperature=temperature)
    return chain.invoke(transcript)