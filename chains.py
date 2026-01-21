import os
from dotenv import load_dotenv
from groq import Groq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env
load_dotenv()

# Read Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def build_index(passages):
    """
    Build a TF-IDF index from document passages.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    matrix = vectorizer.fit_transform(passages)
    return vectorizer, matrix


def answer_question(question, vectorizer, matrix, passages, line_mapping):
    """
    Retrieve the most relevant passage and answer strictly from it.
    """

    # Vectorize the question
    question_vec = vectorizer.transform([question])

    # Compute cosine similarity
    similarities = cosine_similarity(question_vec, matrix)[0]
    best_idx = similarities.argmax()

    context = passages[best_idx]
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