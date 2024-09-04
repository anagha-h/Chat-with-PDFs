# 📄 Chat with PDFs using Google Generative AI

This project is a Streamlit-based web app that allows users to upload PDF documents and ask questions about the content. The app processes the uploaded PDFs, splits the content into manageable chunks, and leverages Google Generative AI to answer user questions based on the document contents.

The project uses LangChain, Google Generative AI, FAISS for text embeddings, and Streamlit for the frontend interface.

## ✨ Features
- **Upload PDFs:** Upload multiple PDF files to extract text from.
- **Conversational AI:** Ask questions about the content of the PDFs, and get detailed answers.
- **Real-time PDF Processing:** Process PDFs and generate embeddings for efficient question answering.
- **Customizable Layout:** The app features a modern and clean UI with a custom CSS option for additional styling.

## 🚀 Getting Started

### Prerequisites
Make sure you have the following installed on your system:
- Python 3.10 or higher
- A Google Cloud API Key for Google Generative AI

### 1. Clone the Repository
```bash
git clone https://github.com/anagha-h/chat-with-pdfs.git
cd chat-with-pdfs
```

### 2. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Set up Environment Variables
Create a `.env` file in the root directory with your Google Cloud API Key:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run the Application
Start the Streamlit app by running the following command:
```bash
streamlit run app.py
```

Once the app is running, open your web browser and go to `http://localhost:8501` to interact with the application.

## 🛠️ How It Works

1. **PDF Upload:** Users can upload multiple PDFs, which are then processed by the app.
2. **Text Extraction:** The app extracts the text from the PDF using the PyPDF2 library.
3. **Text Chunking:** The extracted text is split into manageable chunks using LangChain’s text splitter.
4. **Embedding & FAISS Indexing:** The chunks are embedded using Google Generative AI embeddings, and a FAISS vector store is created for similarity search.
5. **Question Answering:** The user’s question is matched with the most relevant chunks from the PDFs, and the Google Generative AI is used to generate a detailed answer.

## 🌈 Customization

### CSS Customization
You can modify the appearance of the app by editing the `styles.css` file. The CSS file applies custom styles to the Streamlit components, such as buttons, text boxes, headers, etc.

## 📚 Technologies Used
- **Streamlit**: Frontend for building interactive web applications.
- **PyPDF2**: For extracting text from PDF documents.
- **LangChain**: For managing the text chunking and conversational chains.
- **Google Generative AI**: Used for text embeddings and conversational AI model.
- **FAISS**: For efficient similarity search on the document embeddings.