import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path: str):
    """
    Loads a PDF, splits its content into text chunks, and prints the statistics.
    """
    # Check if the PDF file exists before loading
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'.")
        print("Please ensure your PDF is placed in the data/ directory and matches the filename.")
        return

    print(f"Loading PDF from: {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")

    # Initialize the RecursiveCharacterTextSplitter with chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Split the loaded pages into chunks
    chunks = text_splitter.split_documents(pages)
    print(f"Successfully split the document into {len(chunks)} chunks.")

    # Print a sample chunk (the first chunk) if chunks exist
    if chunks:
        print("\n--- Sample Chunk (First Chunk) ---")
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Character Length: {len(chunks[0].page_content)}")
        print("Content:")
        print(chunks[0].page_content)
        print("---------------------------------")
    else:
        print("No chunks were generated.")

if __name__ == "__main__":
    # Path to the PDF file in the data folder
    pdf_path = "data/agentic_ai.pdf"
    process_pdf(pdf_path)
