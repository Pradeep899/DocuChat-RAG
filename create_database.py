# load pdf
# split into chunks
# create the embedding
# store into chroma

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

data = PyPDFLoader(r"D:\ML-Summer\Gen-AI\RAG_Project\Moran_book.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(   # embedding model
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = r"D:\ML-Summer\Gen-AI\RAG_Project\Chroma_db"
)