from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def build_index(passages):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(passages)
    return vectorizer, matrix

def answer_question(question, vectorizer, matrix, passages, line_mapping, max_words=100):
    q_vec = vectorizer.transform([question])
    similarities = cosine_similarity(q_vec, matrix).flatten()
    best_idx = similarities.argmax()
    passage = passages[best_idx]

    sentences = re.split(r'(?<=[.!?])\s+', passage)
    sent_vectors = vectorizer.transform(sentences)
    sent_scores = cosine_similarity(q_vec, sent_vectors).flatten()
    best_sentence = sentences[sent_scores.argmax()]

    words = best_sentence.split()
    answer = " ".join(words[:max_words])

    ref_line = line_mapping.get(best_idx, "N/A")
    return answer, ref_line