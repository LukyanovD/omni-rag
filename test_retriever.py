import sys
import traceback
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.storage import LocalFileStore

emb = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sp = FastEmbedSparse(model_name='Qdrant/bm25')
url = 'http://qdrant:6333'
col = 'test_parent_add'

QdrantVectorStore.from_texts(['test'], embedding=emb, sparse_embedding=sp, retrieval_mode=RetrievalMode.HYBRID, url=url, collection_name=col)

client = QdrantVectorStore(client=None, collection_name=col, embedding=emb, sparse_embedding=sp, retrieval_mode=RetrievalMode.HYBRID, url=url)
docstore = create_kv_docstore(LocalFileStore('/tmp/docstore'))

retriever = ParentDocumentRetriever(
    vectorstore=client,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
)
doc1 = Document(page_content='hello world', metadata={'doc_id': '1'})
try:
    retriever.add_documents([doc1], ids=['1'])
    print('SUCCESS')
except Exception as e:
    traceback.print_exc()
