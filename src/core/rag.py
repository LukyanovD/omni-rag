import os
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from src.core.config import OLLAMA_HOST, OLLAMA_MODEL

def build_rag_chain(base_retriever, top_k=10, temperature=0.0):
    """
    Универсальная функция для создания RAG-цепочки с продвинутыми техниками.
    """
    llm = ChatOllama(
        model=OLLAMA_MODEL, 
        temperature=temperature,
        base_url=OLLAMA_HOST,
        keep_alive="24h"
    )
    
    # Увеличиваем K для первичного поиска, чтобы Reranker'у было из чего выбирать
    base_retriever.search_kwargs = {"k": top_k * 3}
    
    # 1. Query Transformation (Multi-Query Retriever)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )
    
    # 2. Cross-Encoder Reranker
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=top_k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multi_query_retriever
    )
    
    system_prompt = (
        "Ты — эксперт по спортивной фармакологии. Твоя задача — давать развернутые и точные ответы "
        "на вопросы, основываясь на предоставленном контексте. Ты можешь делать аналитические выводы, "
        "опираясь на тексты. Если информации для ответа в контексте совершенно нет, честно скажи: "
        "'В моей базе знаний нет достаточной информации об этом'. Не выдумывай факты и дозировки, "
        "которых нет в источниках. "
        "Отвечай СТРОГО на русском языке. Использование китайского или других языков запрещено."
        "\n\n"
        "Контекст:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, document_chain)
    
    return rag_chain, llm
