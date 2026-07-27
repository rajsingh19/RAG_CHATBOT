import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from the .env file
load_dotenv()

# Print loaded keys to debug environment configuration
print("Loaded API keys from environment:", [k for k in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY"] if k in os.environ])

def process_pdf_and_embed(file_path: str):
    """
    Loads a PDF, splits its content into chunks, and generates embeddings for ALL chunks.
    Uses rate-limit-safe batching (50 chunks per batch) to avoid Gemini Free Tier 429 errors.
    """
    # Check if the PDF file exists before loading
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'.")
        print("Please ensure your PDF is placed in the data/ directory and matches the filename.")
        return None, None

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
        return None, None

    # 3. Generate Embeddings
    print("Generating embeddings for all chunks...")
    
    # Initialize GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Generate embeddings in rate-limit-safe batches of 50
    vector_list = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [chunk.page_content for chunk in batch_chunks]
        
        print(f"Embedding chunks {i + 1} to {min(i + batch_size, len(chunks))} of {len(chunks)}...")
        batch_vectors = embeddings.embed_documents(batch_texts)
        vector_list.extend(batch_vectors)
        
        # Sleep to stay under the Free Tier rate limit of 100 requests per minute
        if i + batch_size < len(chunks):
            print("Sleeping for 15 seconds to avoid Gemini rate limits...")
            time.sleep(15)

    print(f"Successfully generated {len(vector_list)} embeddings.")
    
    if vector_list:
        dimension = len(vector_list[0])
        print(f"Embedding dimension (vector length): {dimension}")
        return chunks, vector_list
    else:
        print("No embeddings were generated.")
        return None, None

def setup_pinecone_index(index_name: str, dimension: int):
    """
    Connects to Pinecone and creates a serverless index if it doesn't already exist.
    """
    # Retrieve the Pinecone API key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key == "your_pinecone_api_key_here":
        print("Error: Please set your PINECONE_API_KEY in the .env file.")
        return None

    # Initialize client
    pc = Pinecone(api_key=api_key)
    
    print(f"Connecting to Pinecone and checking index '{index_name}'...")
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating a new one...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Successfully created index '{index_name}'!")
    else:
        print(f"Index '{index_name}' already exists.")
        
    return pc

def upload_vectors_to_pinecone(pc: Pinecone, index_name: str, chunks: list, vector_list: list):
    """
    Uploads all chunk embeddings to Pinecone with chunk text stored in metadata.
    """
    # Connect to the specific Pinecone index
    index = pc.Index(index_name)
    
    # Prepare the vectors payload with simple numeric IDs and metadata
    vectors_to_upsert = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vector_list)):
        # Numeric ID is the index converted to a string
        vector_id = str(idx)
        
        # Metadata contains the text content of the chunk and its page number
        metadata = {
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", 0)
        }
        
        vectors_to_upsert.append((vector_id, vector, metadata))
        
    print(f"Uploading {len(vectors_to_upsert)} vectors to index '{index_name}'...")
    
    # Upsert vectors in batches to print progress
    upsert_batch_size = 50
    for i in range(0, len(vectors_to_upsert), upsert_batch_size):
        batch = vectors_to_upsert[i : i + upsert_batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded vectors {i + 1} to {min(i + upsert_batch_size, len(vectors_to_upsert))}...")
        
    print(f"Upload complete! Total uploaded vectors: {len(vectors_to_upsert)}")

def query_rag(query: str, index_name: str):
    """
    Embeds a search query, retrieves top 3 matching chunks from Pinecone,
    and uses ChatGoogleGenerativeAI to generate a strictly context-grounded response.
    """
    # 1. Retrieve the Pinecone API key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key == "your_pinecone_api_key_here":
        print("Error: Please set your PINECONE_API_KEY in the .env file.")
        return

    # 2. Initialize Pinecone client and connect to the index
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # 3. Generate embedding for the search query
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    print(f"\nGenerating embedding for query: '{query}'...")
    query_vector = embeddings.embed_query(query)
    
    # 4. Search Pinecone for top 3 matching vectors
    print("Searching Pinecone index...")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    
    matches = results.get("matches", [])
    if not matches:
        print("No matching documents found in Pinecone index.")
        return

    # 5. Extract matching texts for context
    context_chunks = []
    for match in matches:
        text = match.get("metadata", {}).get("text", "")
        if text:
            context_chunks.append(text)
    
    context = "\n\n".join(context_chunks)

    # 6. Initialize ChatGoogleGenerativeAI to generate the answer
    # Using gemini-2.5-flash with temperature 0.0 for deterministic answers
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    
    # Construct strictly grounded system and user instructions
    system_instruction = (
        "You are a helpful assistant. You must answer the user's question ONLY using the provided context. "
        "Do not add any external knowledge or make assumptions. "
        "If the answer is not explicitly present in the provided context, you must reply exactly with: "
        "\"I could not find this information in the provided document.\"\n"
        "Return only the answer."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    # 7. Call LLM to generate answer
    print("Generating answer using Gemini LLM...")
    response = llm.invoke([
        ("system", system_instruction),
        ("human", user_prompt)
    ])
    
    print("\n" + "="*40)
    print("ANSWER:")
    print("="*40)
    print(response.content)
    print("="*40)

if __name__ == "__main__":
    index_name = "rag-chatbot-index"
    pdf_path = "data/agentic_ai.pdf"
    
    # Connect to Pinecone and check status
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key == "your_pinecone_api_key_here":
        print("Error: Please set your PINECONE_API_KEY in the .env file.")
        exit(1)
        
    pc = Pinecone(api_key=api_key)
    
    # Retrieve existing indexes
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    # If the index does not exist, run the setup and ingestion
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Running ingestion pipeline...")
        chunks, vector_list = process_pdf_and_embed(pdf_path)
        if chunks and vector_list:
            dimension = len(vector_list[0])
            pc = setup_pinecone_index(index_name, dimension)
            if pc:
                upload_vectors_to_pinecone(pc, index_name, chunks, vector_list)
    else:
        # Index exists, check if it's empty
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            print(f"Index '{index_name}' exists but is empty. Running ingestion pipeline...")
            chunks, vector_list = process_pdf_and_embed(pdf_path)
            if chunks and vector_list:
                upload_vectors_to_pinecone(pc, index_name, chunks, vector_list)
        else:
            print(f"Index '{index_name}' is already populated with {stats['total_vector_count']} vectors.")

    # Start the interactive querying loop
    print("\nStarting RAG Query & Retrieval Testing Interface. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nEnter search query: ")
            if query.strip().lower() == "exit":
                print("Exiting search interface.")
                break
            if not query.strip():
                continue
            
            # Execute retrieval search and display matches
            query_rag(query.strip(), index_name)
            
        except KeyboardInterrupt:
            print("\nExiting search interface.")
            break
