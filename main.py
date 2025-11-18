# main.py
# AmbedkarGPT - Simple RAG Prototype (LangChain + Chroma + Ollama Mistral)

from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def build_vector_store():
    print("[+] Loading speech.txt ...")
    loader = TextLoader("speech.txt")
    documents = loader.load()

    print("[+] Splitting text into chunks ...")
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print("[+] Creating embeddings ...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("[+] Storing vectors in ChromaDB ...")
    vectordb = Chroma.from_documents(chunks, embeddings, collection_name="speech_store")

    return vectordb


def create_qa_chain(vectordb):
    print("[+] Initializing Ollama (Mistral)...")
    llm = Ollama(model="mistral")

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain


def main():
    print("=== AmbedkarGPT: Command-Line Q&A System ===")

    vectordb = build_vector_store()
    qa = create_qa_chain(vectordb)

    print("\nAsk any question related to the speech.")
    print("Type 'exit' to quit.\n")

    while True:
        user_q = input("You: ")

        if user_q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        print("\nThinking...\n")
        answer = qa.run(user_q)
        print("AmbedkarGPT:", answer)
        print("\n" + "-"*40 + "\n")


if __name__ == "__main__":
    main()
