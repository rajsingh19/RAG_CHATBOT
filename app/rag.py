import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from the .env file
load_dotenv()

# Print loaded keys to debug environment configuration
print("Loaded API keys from environment:", [k for k in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY"] if k in os.environ])

def process_pdf_and_embed(file_path: str):
    """
    Loads a PDF, splits its content into chunks, and generates embeddings for all chunks using Gemini API.
    """
    # Check if the PDF file exists before loading
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'.")
        print("Please ensure your PDF is placed in the data/ directory and matches the filename.")
        return None

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
        return None

    # 3. Generate Embeddings
    print("Generating embeddings for all chunks (this requires GOOGLE_API_KEY in .env)...")
    
    # Initialize GoogleGenerativeAIEmbeddings
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
        print(f"First embedding sample (first 5 values): {vector_list[0][:5]}")
        return dimension
    else:
        print("No embeddings were generated.")
        return None

def setup_pinecone_index(index_name: str, dimension: int):
    """
    Connects to Pinecone and creates a serverless index if it doesn't already exist.
    """
    # Retrieve the Pinecone API key from environment variables
    api_key = os.environ.get("PINECONE_API_KEY")
    
    # Validate that the API key has been set and isn't the placeholder
    if not api_key or api_key == "your_pinecone_api_key_here":
        print("Error: Please set your PINECONE_API_KEY in the .env file.")
        return

    # 1. Initialize the official Pinecone client with your API key
    pc = Pinecone(api_key=api_key)
    
    print(f"Connecting to Pinecone and checking index '{index_name}'...")
    
    # 2. Fetch the list of existing index names in your Pinecone account
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    # 3. If the index does not already exist, create it
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating a new one...")
        
        # Create a new serverless index with cosine similarity metric
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",  # Requirements: Use cosine similarity
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Standard region for the Pinecone free tier
            )
        )
        print(f"Successfully created index '{index_name}'!")
    else:
        print(f"Index '{index_name}' already exists.")

if __name__ == "__main__":
    # Path to the PDF file in the data folder
    pdf_path = "data/agentic_ai.pdf"
    
    # Process PDF and get the embedding dimension
    dimension = process_pdf_and_embed(pdf_path)
    
    if dimension:
        # Define index name
        index_name = "rag-chatbot-index"
        # Connect to Pinecone and set up the index
        setup_pinecone_index(index_name, dimension)
