from click import prompt
from dotenv import load_dotenv
load_dotenv()

# from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model = HuggingFaceEmbeddings(   # embedding model
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma(
    embedding_function = embedding_model,
    persist_directory = r"D:\ML-Summer\Gen-AI\RAG_Project\Chroma_db"
)

retriver = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k" : 10,   # out of 10, select 4
        "lambda_mult" : 0.5     # diversity 
    }
)

# llm = ChatMistralAI(
#     model = "mistral-small-2506"
# )
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)

# prompt template
template = ChatPromptTemplate.from_messages(
    [
        ("system","""
        You are a helpful AI assistant. You will be provided with a document and a question.
        Your task is to answer the question based on the document provided.
        Use only the provided context to answer the question. If the answer is not contained within the text below,
        say "I couldn't find the answer in the provided document""" 
        ),
        ("human","""
        context : {context},
        question : {question}"""
        )
    ]
)

print("------------------------------------------------------------")
print("RAG System created")
print("press 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break
    docs = retriver.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt = template.invoke(
        {
            "context" : context,
            "question" : query
        }
    )
    
    response = llm.invoke(final_prompt)
    if isinstance(response.content, str):
        answer_text = response.content
    else:
        answer_text = "".join(
            block.get("text", "") for block in response.content if isinstance(block, dict)
        )
    
    print(f"\n AI : {answer_text}")