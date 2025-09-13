import streamlit as st
from utils import preprocess_markdown, preprocess_text, preprocess_pdf, load_chat_log, save_chat_log
from chains import build_index, answer_question

st.set_page_config(page_title="Document Q&A Chatbot", layout="wide")
st.title("Document Q&A Chatbot")

# load chat log
chat_log = load_chat_log()

# upload area (allow md, txt, pdf)
uploaded_file = st.file_uploader("Upload a document (.md, .txt, .pdf)", type=["md", "txt", "pdf"])

# session storage
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None
if "matrix" not in st.session_state:
    st.session_state["matrix"] = None
if "passages" not in st.session_state:
    st.session_state["passages"] = []
if "line_mapping" not in st.session_state:
    st.session_state["line_mapping"] = {}

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "md":
        raw_text = uploaded_file.read().decode("utf-8")
        passages, line_mapping = preprocess_markdown(raw_text)
    elif file_type == "txt":
        raw_text = uploaded_file.read().decode("utf-8")
        passages, line_mapping = preprocess_text(raw_text)
    elif file_type == "pdf":
        passages, line_mapping = preprocess_pdf(uploaded_file)
    else:
        passages, line_mapping = [], {}

    if passages:
        vectorizer, matrix = build_index(passages)
        st.session_state["vectorizer"] = vectorizer
        st.session_state["matrix"] = matrix
        st.session_state["passages"] = passages
        st.session_state["line_mapping"] = line_mapping
        st.success(f"{file_type.upper()} document uploaded and indexed successfully.")
    else:
        st.error("Could not process the uploaded file. Please check the format.")
else:
    st.info("Please upload a Markdown, Text, or PDF file to start.")

# question input
question = st.text_input("Enter your question about the uploaded document:")

if st.button("Ask"):
    if uploaded_file is None:
        st.error("Please upload a document first.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        vectorizer = st.session_state.get("vectorizer")
        matrix = st.session_state.get("matrix")
        passages = st.session_state.get("passages")
        line_mapping = st.session_state.get("line_mapping")

        if vectorizer is None or matrix is None:
            st.error("Document not indexed yet. Re-upload or wait a moment.")
        else:
            answer, ref_line = answer_question(
                question, vectorizer, matrix, passages, line_mapping
            )

            # save to chat log
            chat_log.append({"question": question, "answer": answer, "reference": ref_line})
            chat_log = chat_log[-5:]
            save_chat_log(chat_log)

            # show result
            st.markdown("**Answer:**")
            st.write(answer)
            st.markdown(f"**Reference line:** {ref_line}")

# display history
if chat_log:
    st.markdown("### Recent chat history (last 5)")
    for i, e in enumerate(chat_log[-5:], 1):
        st.markdown(f"**Q{i}:** {e['question']}")
        st.markdown(f"**A{i}:** {e['answer']}")
        st.markdown(f"Reference line: {e['reference']}")
        st.markdown("---")