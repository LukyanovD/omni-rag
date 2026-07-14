from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.core.config import OLLAMA_HOST, OLLAMA_MODEL

def build_rag_chain(base_retriever, top_k=4):
    llm = ChatOllama(
        model=OLLAMA_MODEL, 
        temperature=0.0,
        base_url=OLLAMA_HOST,
        keep_alive="24h"
    )

    from langchain_core.prompts import PromptTemplate
    
    prompt = PromptTemplate.from_template(
        "Используй ТОЛЬКО следующий текст для ответа на вопрос. "
        "Если в тексте нет ответа, напиши 'В базе данных нет информации по этому вопросу'. "
        "Категорически запрещено выдумывать факты или использовать свои знания.\n\n"
        "ТЕКСТ:\n{context}\n\n"
        "ИСТОРИЯ ДИАЛОГА:\n{chat_history}\n\n"
        "ВОПРОС: {input}\n\n"
        "ОТВЕТ:"
    )
    
    document_chain = create_stuff_documents_chain(
        llm, 
        prompt
    )
    
    # Инициализируем сверхлегкий мультиязычный кросс-энкодер
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    compressor = CrossEncoderReranker(model=model, top_n=top_k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    
    rag_chain = create_retrieval_chain(compression_retriever, document_chain)
    
    return rag_chain, llm
