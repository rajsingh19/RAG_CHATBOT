import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Print loaded keys for debugging
print("Loaded API keys from environment:", [k for k in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"] if k in os.environ])


def process_pdf_and_embed(file_path: str):
    """
    Loads a PDF, splits its content into chunks, and generates embeddings for all chunks using Gemini API.
    """
    # Check if the PDF file exists before loading
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'.")
        print("Please ensure your PDF is placed in the data/ directory and matches the filename.")
        return

    # 1. Load PDF
    print(f"Loading PDF from: {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")

    # 2. Split PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Successfully split into {len(chunks)} chunks.")

    if not chunks:
        print("No chunks to embed.")
        return

    # 3. Generate Embeddings
    print("Generating embeddings for all chunks (this requires GOOGLE_API_KEY in .env)...")
    
    # Initialize GoogleGenerativeAIEmbeddings (reads GOOGLE_API_KEY from env automatically)
    # Using the standard 'models/gemini-embedding-2' model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Extract text content from each chunk (limit to 50 to avoid Gemini Free Tier 100 RPM rate limit)
    texts = [chunk.page_content for chunk in chunks][:50]
    print(f"Embedding the first {len(texts)} chunks to stay under Gemini free-tier rate limits (100 RPM)...")
    
    # Generate the embeddings
    vector_list = embeddings.embed_documents(texts)
    print(f"Successfully generated {len(vector_list)} embeddings.")

    # Print validation information
    if vector_list:
        dimension = len(vector_list[0])
        print(f"Embedding dimension (vector length): {dimension}")
        print(f"First embedding length: {len(vector_list[0])}")
        # Print a small sample of the first vector to prove it's a list of floats
        print(f"First embedding sample (first 5 values): {vector_list[0][:5]}")
    else:
        print("No embeddings were generated.")

if __name__ == "__main__":
    # Path to the PDF file in the data folder
    pdf_path = "data/agentic_ai.pdf"
    process_pdf_and_embed(pdf_path)
