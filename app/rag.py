import os
from langchain_community.document_loaders import PyPDFLoader

def load_agentic_pdf(file_path: str):
    """
    Loads a PDF file using PyPDFLoader, prints the total page count,
    and displays the content of the first page.
    """
    # Check if the PDF file exists before loading
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'.")
        print("Please ensure your PDF is placed in the correct directory and matches the filename.")
        return

    print(f"Loading PDF from: {file_path}...")
    
    # Initialize the PyPDFLoader
    loader = PyPDFLoader(file_path)
    
    # Load all pages of the document
    pages = loader.load()
    
    # Print the total number of pages loaded
    print(f"Successfully loaded {len(pages)} pages.")
    
    # Print the content of the first page (index 0) if pages exist
    if pages:
        print("\n--- First Page Content ---")
        print(pages[0].page_content)
        print("---------------------------")
    else:
        print("The PDF file is empty.")

if __name__ == "__main__":
    # Define the path to the Agentic AI PDF in the data directory
    pdf_path = "data/agentic_ai.pdf"
    
    load_agentic_pdf(pdf_path)
