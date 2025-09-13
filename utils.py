import re
import json
import os
import PyPDF2

CHAT_LOG_FILE = "chat_log.json"

def preprocess_markdown(md_text, chunk_lines=4):
    """Convert markdown text into passages + line mapping."""
    raw_lines = [ln.rstrip() for ln in md_text.splitlines()]
    cleaned_lines = []
    for ln in raw_lines:
        ln_clean = re.sub(r"\!\[.*?\]\(.*?\)", "", ln)      # remove images
        ln_clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", ln_clean)  # keep link text
        ln_clean = re.sub(r"[#>*`-]{1,}", "", ln_clean)     # remove markdown tokens
        ln_clean = ln_clean.strip()
        if ln_clean:
            cleaned_lines.append(ln_clean)
    return chunk_lines_into_passages(cleaned_lines, chunk_lines)

def preprocess_text(txt_text, chunk_lines=4):
    """Process plain text files into passages."""
    raw_lines = [ln.rstrip() for ln in txt_text.splitlines() if ln.strip()]
    return chunk_lines_into_passages(raw_lines, chunk_lines)

def preprocess_pdf(uploaded_file, chunk_lines=4):
    """Extract text from PDF and process into passages."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        raw_lines = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_lines.extend(text.splitlines())
        cleaned = [ln.strip() for ln in raw_lines if ln.strip()]
        return chunk_lines_into_passages(cleaned, chunk_lines)
    except Exception as e:
        print("Error reading PDF:", e)
        return [], {}

def chunk_lines_into_passages(lines, chunk_lines=4):
    """Group lines into passages, preserving Q/A blocks."""
    passages = []
    line_mapping = {}
    buffer = []
    for idx, line in enumerate(lines, 1):
        if line.strip() == "":
            if buffer:
                passages.append(" ".join(buffer))
                line_mapping[len(passages) - 1] = idx - len(buffer)
                buffer = []
        else:
            buffer.append(line.strip())
    if buffer:
        passages.append(" ".join(buffer))
        line_mapping[len(passages) - 1] = len(lines) - len(buffer) + 1
    return passages, line_mapping

def load_chat_log():
    if os.path.exists(CHAT_LOG_FILE):
        try:
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_log(chat_log):
    last_five = chat_log[-5:]
    with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(last_five, f, ensure_ascii=False, indent=2)