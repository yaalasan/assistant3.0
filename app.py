from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline


# Load model ONCE at module startup to avoid timeouts on Render
print("Loading LLM model (this may take a minute on first load)...")
_hf_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    temperature=0.0,
    device=-1  # Use CPU; set to 0 for GPU if available
)
_llm = HuggingFacePipeline(pipeline=_hf_pipe)
print("LLM model loaded successfully")


#core logic
def build_qa(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #create FAISS db
    db = FAISS.from_documents(chunks, embed)

    # Use cached LLM instead of loading it every time
    llm = _llm

    # Create a retrieval chain
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # Create prompt template for the final answer
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the chain
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return qa_chain




