# ChatPDF  
![ChatPDF Screenshot](ChatPDF/tree/main/Image.png)

## 🚀 Overview  
*ChatPDF* is a web-based application that allows you to upload PDF documents and query them using natural-language questions. Behind the scenes, the system extracts and processes text from PDFs, converts them into embeddings, and uses a retrieval-augmented generation (RAG) framework to provide conversational answers.  
This repository contains the frontend (TypeScript / React) and backend (Python) modules necessary to build and run the application locally or deploy it.

## 🔍 Features  
- Upload single or multiple PDF files via a user-friendly interface.  
- Automatic text extraction and chunking of PDF content.  
- Embedding generation and vector search for rapid retrieval of relevant content.  
- Natural-language chat interface to ask questions and receive responses grounded in the uploaded PDFs.  
- Interactive UI that shows conversation history and allows you to clear context or re-load documents.  
- Modular architecture: you can swap embedding models, adjust chunking logic, or customize retrieval strategies.

## 🧱 Architecture & Tech Stack  
**Frontend**:  
- React + TypeScript  
- Vite or similar build tooling  
- UI components for file upload, chat interface, and context display  

**Backend**:  
- Python (FastAPI / Flask / whichever you use)  
- PDF text extraction (e.g., PyPDF2 or pdfminer)  
- Embedding generation (OpenAI embeddings or other model)  
- Vector database (FAISS / Chroma / Pinecone) for retrieval  
- Chat endpoint to combine user query + context + model generation  

## 📁 Project Structure  
/
├── backend/ # Python backend code for processing, embedding & API
├── components/ # Frontend React components (chat UI, upload widget, etc.)
├── services/ # Shared services (e.g., API client, embedding wrapper)
├── Image/ # Folder containing screenshots & UI images
│ └── your-screenshot-filename.png
├── App.tsx # Entry point for the frontend
├── index.tsx # Frontend bootstrap file
├── package.json # Frontend dependencies
├── requirements.txt # Backend Python dependencies
└── README.md # This file


## 🏁 Getting Started  
### Prerequisites  
- Node.js (version >= 16) installed  
- Python (version >= 3.8) installed  
- (Optional) An API key if you're using a hosted embedding or language model service  

### Setup & Run  
**1. Clone the repository**
```bash
git clone https://github.com/Amanjha112113/ChatPDF.git
cd ChatPDF

**2. Backend setup**
cd backend
pip install -r requirements.txt
# Set required environment variables, e.g.:
# export OPENAI_API_KEY="your_key_here"
# export VECTOR_DB_URL="your_vector_db_connection"
# etc.
python main.py  # or however the backend is started


**3. Frontend setup**
cd ../
npm install
npm run dev


**4. Open your browser**
Visit http://localhost:3000 (or the port configured) to use ChatPDF. Upload a PDF and start chatting!
