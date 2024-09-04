import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

st.set_page_config(page_title="Chat with PDFs", page_icon="📄", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")  

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question with as many details as possible from the provided context.
    If the answer is not available in the context, say "The answer is not available in the context."
    
    Context:\n{context} \n
    Question:\n{question} \n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    st.write("**Reply**:", response.get("output_text", "No answer available."))

def main():
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📄 Chat with your PDFs using Gemini!</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.sidebar.title("Menu")
        pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", type="pdf", accept_multiple_files=True)  
        if st.sidebar.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():  
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete")
                else:
                    st.error("No text found in the uploaded PDFs")
    
    with col2:
        st.markdown("""
        <div style='padding: 15px; background-color: #F0F2F6; border-radius: 10px;'>
        <h3 style='color: #0D47A1;'>Chat with the processed PDF files</h3>
        <p style='color: #4A4A4A;'>Use the input box below to ask a question about the content of your uploaded PDFs, and get a response based on the extracted information.</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_question = st.text_input("Ask a question about the PDF files", placeholder="Type your question here...")
        
        if st.button("Get Answer", key="question"):
            if user_question:
                user_input(user_question)
            else:
                st.error("Please ask a question.")
    
    st.markdown("""
    <hr>
    <p style='text-align: center; color: #7F7F7F;'>Made with 💻 by Anagha Honnali</p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
