from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.core.config import OLLAMA_HOST, OLLAMA_MODEL

def build_rag_chain(base_retriever, top_k=10):
    llm = ChatOllama(
        model=OLLAMA_MODEL, 
        temperature=0.0,
        base_url=OLLAMA_HOST,
        keep_alive="24h"
    )

    system_prompt = (
        "Ты — точный ИИ-редактор по спортивной медицине.\n"
        "Тебе даны фрагменты текста <context>. Твоя задача — ответить на вопрос, основываясь ТОЛЬКО на этих фрагментах.\n\n"
        "ПРАВИЛА:\n"
        "1. Изучи фрагменты и найди в них информацию, относящуюся к вопросу.\n"
        "2. Сформулируй ответ связным, понятным и литературным русским языком. Текст должен читаться легко.\n"
        "3. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО придумывать факты, детали или причины, которых нет в тексте. Опирайся строго на прочитанное.\n"
        "4. Если ответа в тексте нет, ответь: \"В базе данных нет информации по этому вопросу.\""
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "<context>\n{context}\n</context>\n\nВопрос: {input}\n\nОтвет:"),
    ])
    
    document_chain = create_stuff_documents_chain(
        llm, 
        prompt
    )
    
    retriever_chain = base_retriever
    
    rag_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    return rag_chain, llm
