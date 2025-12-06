from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from huggingface_hub import InferenceApi
import os


# Load model ONCE at module startup to avoid timeouts on Render
print("Note: using Hugging Face Inference API for remote generation (no large local model will be loaded).")
HF_MODEL = os.environ.get("HF_MODEL", "google/flan-t5-base")

# Create the Inference API client at startup; requires HUGGINGFACEHUB_API_TOKEN in environment
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    hf_client = InferenceApi(repo_id=HF_MODEL, token=hf_token)
else:
    hf_client = None


#core logic
def build_qa(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #create FAISS db
    db = FAISS.from_documents(chunks, embed)

    # Create a retriever from the FAISS index
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Prompt template string
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # A small QA wrapper that uses the HF Inference API for generation
    class RemoteQA:
        def __init__(self, retriever, prompt_template, hf_client):
            self.retriever = retriever
            self.prompt_template = prompt_template
            self.hf_client = hf_client

        def _get_docs(self, question):
            # support different retriever method names
            if hasattr(self.retriever, "get_relevant_documents"):
                return self.retriever.get_relevant_documents(question)
            if hasattr(self.retriever, "retrieve"):
                return self.retriever.retrieve(question)
            # fallback: empty
            return []

        def invoke(self, question: str):
            if self.hf_client is None:
                return "Hugging Face token not configured on the server. Set HUGGINGFACEHUB_API_TOKEN."

            docs = self._get_docs(question)
            context = format_docs(docs)
            prompt_text = self.prompt_template.format(context=context, question=question)

            try:
                # call the HF Inference API (adjust params as needed)
                res = self.hf_client(prompt_text, parameters={"max_new_tokens": 256})

                # extract text robustly from the API response
                if isinstance(res, dict):
                    text = res.get("generated_text") or res.get("text") or str(res)
                elif isinstance(res, list):
                    # sometimes returns a list of dicts
                    item = res[0] if res else {}
                    if isinstance(item, dict):
                        text = item.get("generated_text") or item.get("text") or str(item)
                    else:
                        text = str(item)
                else:
                    text = str(res)

                return text
            except Exception as e:
                return f"Error during HF inference: {e}"

    return RemoteQA(retriever, template, hf_client)




