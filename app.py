import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
import os


# Sidebar contents
with st.sidebar:
    st.title('😎💬➡️ LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [My Linkedin](https://www.linkedin.com/in/gautham-bharati/)
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Hugging Face](https://huggingface.co/) LLM models
 
    ''')

    st.write('Made by [Gautham Bharati]') 

def main():
    st.header('Chat with PDF')

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # Use Hugging Face model for QA
            qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
            context = " ".join([doc.page_content for doc in docs])
            response = qa_model(f"Question: {query}\n\nContext: {context}")
            st.write(response[0]['generated_text'])

if __name__ == '__main__':
    main()
    
    