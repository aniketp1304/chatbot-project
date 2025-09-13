from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_index(passages):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(passages)
    return vectorizer, matrix

def answer_question(question, vectorizer, matrix, passages, line_mapping, max_words=100):
    # Vectorize the question
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, matrix).flatten()
    best_idx = sims.argmax()
    best_passage = passages[best_idx]

    # If passage contains Q→A, return the answer part
    if "Q:" in best_passage and "A:" in best_passage:
        parts = best_passage.split("A:", 1)
        if len(parts) == 2:
            answer_text = parts[1].strip()
        else:
            answer_text = best_passage.strip()
    else:
        answer_text = best_passage.strip()

    # Limit word count
    words = answer_text.split()
    if len(words) > max_words:
        answer_text = " ".join(words[:max_words]) + "..."

    # Get reference line
    ref_line = line_mapping.get(best_idx, "?")

    return answer_text, ref_line