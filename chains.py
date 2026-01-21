def answer_question(question, vectorizer, matrix, passages, line_mapping):
    """
    Retrieve the most relevant passage and answer strictly from it.
    """

    # Vectorize the question
    question_vec = vectorizer.transform([question])

    # Compute cosine similarity
    similarities = cosine_similarity(question_vec, matrix)[0]
    best_idx = similarities.argmax()

    # Limit context size to avoid 413 error
    context = passages[best_idx][:1500]  # 🔴 IMPORTANT LINE
    reference_line = line_mapping.get(best_idx, "N/A")

    prompt = f"""
You are a document-based question answering assistant.

RULES:
- Answer ONLY using the information in the context.
- If the answer is not in the context, say:
  "The answer is not available in the provided document."
- Keep the answer under 100 words.
- Do NOT add external knowledge.

Context:
\"\"\"
{context}
\"\"\"

Question:
{question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
        return answer, reference_line

    except Exception:
        return (
            "The document section is too large to process. Please ask a more specific question.",
            reference_line
        )