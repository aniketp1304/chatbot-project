Document Q&A Chatbot

A Streamlit-based document question answering chatbot that allows users to upload PDF, TXT, or Markdown files and ask questions answered strictly from the document content.
The system uses TF-IDF semantic retrieval and Groq’s LLaMA-3.1 model to generate accurate, document-grounded answers without hallucination.

⸻

Features
	•	Upload PDF, TXT, and Markdown documents
	•	Ask natural language questions about the document
	•	Answers are generated only from the uploaded content
	•	Displays reference line for traceability
	•	Maintains last 5 question–answer interactions
	•	Secure API key handling (no hardcoded secrets)

⸻

Tech Stack
	•	Python
	•	Streamlit (UI)
	•	scikit-learn (TF-IDF, cosine similarity)
	•	Groq API (LLaMA-3.1-8B-Instant)
	•	PyPDF2
	•	Streamlit Community Cloud (deployment)

⸻

How It Works
	1.	User uploads a document
	2.	Document is preprocessed and split into passages
	3.	TF-IDF is used to find the most relevant passage
	4.	The passage is sent to Groq LLaMA-3.1
	5.	The model answers only using the document context

⸻

Project Structure

chatbot-project/
├── main.py
├── chains.py
├── utils.py
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md

Local Setup

git clone https://github.com/aniketp1304/chatbot-project.git
cd chatbot-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Create a .env file:

GROQ_API_KEY=your_groq_api_key_here

Run the app:

python3 -m streamlit run main.py

Deployment

The application is deployed on Streamlit Community Cloud using GitHub integration and secure secret management.

⸻

Author

Aniket Palsodkar

# chatbot-project
Youtube shorts link for the tutorial->>> https://youtube.com/shorts/1rjv1sHLzXA?si=C4210hO31jclaDZP
