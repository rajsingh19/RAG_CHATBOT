import streamlit as st
import requests

# Set page config
st.set_page_config(page_title="RAG Chatbot", layout="centered")

# Page title and short description
st.title("📚 RAG Chatbot Interface")
st.write("Ask questions and get answers grounded directly in the uploaded PDF document.")

# Input field for user question
question = st.text_input("Enter your question:", placeholder="e.g. What is Agentic AI?")

# Send button to trigger API call
if st.button("Send"):
    if not question.strip():
        st.warning("Please type a question before sending.")
    else:
        # Backend FastAPI chat endpoint URL
        api_url = "http://127.0.0.1:8001/chat"
        payload = {"question": question.strip()}
        
        # Show loading spinner while waiting for response
        with st.spinner("Retrieving context and generating answer..."):
            try:
                response = requests.post(api_url, json=payload, timeout=60)
                
                # Check for successful response
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    context_chunks = data.get("context", [])
                    confidence = data.get("confidence", 0.0)
                    
                    # 1. Out-of-Context Response Check
                    # If similarity score is very low (less than 0.50) or context is empty, override answer
                    is_out_of_context = not context_chunks or confidence < 0.50
                    if is_out_of_context:
                        answer = "I couldn't find this information in the uploaded PDF."
                    
                    # 2. Answer Section
                    # Display answer inside a clean container with heading
                    with st.container():
                        st.markdown("## 🤖 Answer")
                        st.write(answer)
                    
                    # 3. Confidence level and coloring
                    confidence_percentage = confidence * 100
                    if confidence_percentage >= 80:
                        st.success(f"Confidence: {confidence_percentage:.2f}% (High)")
                    elif confidence_percentage >= 60:
                        st.info(f"Confidence: {confidence_percentage:.2f}% (Medium)")
                    else:
                        st.warning(f"Confidence: {confidence_percentage:.2f}% (Low)")
                    
                    # 4. Retrieved Context Section
                    st.markdown("### 📄 Retrieved Context")
                    if not is_out_of_context:
                        for idx, chunk in enumerate(context_chunks):
                            score = chunk.get("score", 0.0)
                            text = chunk.get("text", "")
                            
                            # Trim text to first 300 characters
                            trimmed_text = text[:300] + "..." if len(text) > 300 else text
                            
                            # Title of expander. Since page metadata is not present in our API response,
                            # we don't display anything extra as per requirements.
                            expander_title = f"Retrieved Context {idx + 1}"
                            
                            with st.expander(expander_title):
                                st.write(f"**Similarity Score:** {score * 100:.2f}%")
                                st.write(trimmed_text)
                    else:
                        st.write("No matching context chunks available.")
                    
                    # 5. Footer (displayed at the bottom of every response)
                    st.write("---")
                    st.caption("Source: Answer generated only from the uploaded PDF using Retrieval-Augmented Generation (RAG).")
                else:
                    # Handle API status errors
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                # Handle connection issues (e.g. backend down)
                st.error("Could not connect to the FastAPI backend. Please verify uvicorn is running on http://127.0.0.1:8001")
            except Exception as e:
                # Handle other unexpected exceptions
                st.error(f"An unexpected error occurred: {str(e)}")
