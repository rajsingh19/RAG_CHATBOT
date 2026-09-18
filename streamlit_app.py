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
                    retrieval_score = data.get("retrieval_score", data.get("confidence", 0.0))
                    answer_grounded = data.get("answer_grounded", True)
                    answer_confidence = data.get("answer_confidence", "High")
                    
                    is_abstention = "could not find this information in the provided document" in answer.lower()
                    
                    # 1. Answer Section
                    with st.container():
                        st.markdown("## 🤖 Answer")
                        st.write(answer)
                    
                    # 2. Status & Confidence Indicators
                    if is_abstention or not answer_grounded:
                        st.warning("⚠️ Information Not Found in Document (Abstained)")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            if answer_confidence == "High":
                                st.success("✅ Grounding: High Confidence")
                            elif answer_confidence == "Medium":
                                st.info("ℹ️ Grounding: Medium Confidence")
                            else:
                                st.warning("⚠️ Grounding: Low Confidence")
                        with col2:
                            st.metric(label="Retrieval Similarity Score", value=f"{retrieval_score:.4f}")
                    
                    # 3. Retrieved Context Section
                    st.markdown("### 📄 Retrieved Context Evidence")
                    if context_chunks:
                        for idx, chunk in enumerate(context_chunks):
                            score = chunk.get("score", 0.0)
                            text = chunk.get("text", "")
                            sec = chunk.get("section", "General")
                            ch = chunk.get("chapter", "")
                            pdf_p = chunk.get("pdf_page_number", "N/A")
                            prt_p = chunk.get("printed_page_number", "N/A")
                            c_id = chunk.get("chunk_id", f"chunk_{idx}")
                            
                            expander_title = f"Context {idx + 1}: {sec} (PDF Page {pdf_p}, Printed Page {prt_p})"
                            
                            with st.expander(expander_title, expanded=(idx == 0)):
                                st.markdown(f"**Section:** {sec} | **Chapter:** {ch}")
                                st.markdown(f"**Physical PDF Page:** {pdf_p} | **Printed Page:** {prt_p} | **Retrieval Score:** {score:.4f}")
                                st.text_area("Content:", value=text, height=180, key=f"chunk_txt_{idx}")
                    else:
                        st.write("No context chunks retrieved.")
                    
                    # 4. Footer
                    st.write("---")
                    st.caption("Source: Answer generated exclusively from the uploaded PDF using Structure-Aware RAG.")
                else:
                    # Handle API status errors
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                # Handle connection issues (e.g. backend down)
                st.error("Could not connect to the FastAPI backend. Please verify uvicorn is running on http://127.0.0.1:8001")
            except Exception as e:
                # Handle other unexpected exceptions
                st.error(f"An unexpected error occurred: {str(e)}")
