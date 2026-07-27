import streamlit as st
import requests

# Set page title and layout
st.set_page_config(page_title="RAG Chatbot", layout="centered")

# Page title and short description
st.title("📚 RAG Chatbot Interface")
st.write("Ask questions and get answers grounded directly in the uploaded PDF document.")

# Chat input field for typing questions
question = st.text_input("Enter your question:", placeholder="e.g. What is Agentic AI?")

# Send button to trigger the FastAPI /chat call
if st.button("Send"):
    if not question.strip():
        st.warning("Please type a question before sending.")
    else:
        # FastAPI backend endpoint
        api_url = "http://127.0.0.1:8001/chat"
        payload = {"question": question.strip()}
        
        # Show loading spinner while calling the API
        with st.spinner("Retrieving context and generating answer..."):
            try:
                response = requests.post(api_url, json=payload, timeout=60)
                
                # Check for successful response
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    context_chunks = data.get("context", [])
                    confidence = data.get("confidence", 0.0)
                    
                    # Display the assistant's answer in a clean chat message format
                    st.chat_message("assistant").write(answer)
                    
                    # Display the overall confidence score below the answer
                    st.write(f"**Confidence:** `{confidence:.4f}`")
                    
                    # Expandable section for retrieved context chunks
                    with st.expander("Retrieved Context"):
                        if not context_chunks:
                            st.write("No context chunks retrieved.")
                        else:
                            for idx, chunk in enumerate(context_chunks):
                                score = chunk.get("score", 0.0)
                                text = chunk.get("text", "")
                                
                                # Display each retrieved chunk separately with its score
                                st.write(f"**Chunk {idx + 1} (Similarity Score: {score:.4f})**")
                                st.info(text)
                                st.write("---")
                else:
                    # Handle API status errors
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                # Handle connection issues (e.g. backend down)
                st.error("Could not connect to the FastAPI backend. Please verify uvicorn is running on http://127.0.0.1:8001")
            except Exception as e:
                # Handle other unexpected exceptions
                st.error(f"An unexpected error occurred: {str(e)}")
