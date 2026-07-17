# IIIT Bhopal AI Assistant 🎓

Every semester, the same questions flood the college admin office and faculty inboxes — fee deadlines, hostel allotment, placement cutoffs. Students wait on email replies or dig through scattered PDFs for basic info.

This is a RAG chatbot that fixes that: it's trained on IIIT Bhopal's own website and documents, and answers instantly — 24/7, no waiting.
<!-- put your screenshot at docs/screenshot.png -->
![Screenshot](docs/screenshot.png(2))

---

## How it works

1. A crawler walks `iiitbhopal.ac.in`, pulls down PDFs (fee structures, circulars, notices) and page text
2. That content gets chunked and embedded into a Chroma vector database
3. When a user asks something, the most relevant chunks get retrieved and handed to Gemini as context
4. Gemini answers using that context — or falls back to normal general knowledge if the question isn't college-related

```
User → Flask API → Vector Search (Chroma) → Gemini 2.5 Flash → Answer
```

---

## Tech Stack

Flask · LangChain · Gemini 2.5 Flash · Chroma DB · BeautifulSoup

---

## Running it locally

```bash
git clone https://github.com/Navadeep73/iiitbhopal-ai-assistant.git
cd your-repo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Add your Gemini key to a `.env` file:
```
GOOGLE_API_KEY=your_key_here
```

Then:
```bash
python inges.py     # crawl the site, build the knowledge base
python app.py        # start the server → localhost:5000
```

---

## API

**POST** `/ask`
```json
{ "query": "What are the hostel fees?" }
```
```json
{ "answer": "...", "used_rag": true }
```

---

## Why this matters

This isn't just a demo — it's built to actually be pitched to the college. The plan: get access to internal notices and circulars, host it on an official subdomain, and let it genuinely cut down the repetitive-query load on staff instead of sitting as a portfolio project.

---

