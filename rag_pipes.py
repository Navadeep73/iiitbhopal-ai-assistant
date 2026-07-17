import os
import sys

from dotenv import load_dotenv
load_dotenv()

from inges import load_all_documents

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage




GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()

if not GOOGLE_API_KEY:
    print(
        "\n  GOOGLE_API_KEY is not set.\n"
        "    Add it to your .env file:\n\n"
        "        GOOGLE_API_KEY=your_key_here\n"
    )
    sys.exit(1)


print("\n" + "═"*55)
print("    IIIT Bhopal Assistant — Starting Up")
print("═"*55)

print("\n[1/5] 📂  Loading documents…")
documents = load_all_documents()

if not documents:
    print(
        "\n  No documents found in data/.\n"
        "    Run ingestion first:\n\n"
        "        python inges.py\n"
    )
    sys.exit(1)

print(f"        {len(documents)} document(s) loaded.")




print("\n[2/5]    Splitting into chunks…")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # slightly smaller → more focused chunks
    chunk_overlap=200,
)
docs = splitter.split_documents(documents)
print(f"        {len(docs)} chunk(s) created.")




print("\n[3/5]   Building vector store (this takes a minute first time)…")

# Gemini's hosted embedding API — no local model weights, no torch,
# keeps the process light enough for free-tier hosting (Render 512MB etc.)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="vector_db",
)

# k=5 directly — no local cross-encoder reranker (that model alone was
# ~90MB + torch runtime, the biggest chunk of the old memory footprint)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
print("        Vector store ready.")




chat_history = InMemoryChatMessageHistory()


def format_history(history: InMemoryChatMessageHistory) -> str:
    lines = []
    for msg in history.messages[-6:]:          
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines) if lines else "None"



PROMPT = PromptTemplate(
    template="""You are a brilliant, friendly AI assistant with these superpowers:
1. Deep knowledge about IIIT Bhopal (use the context below when relevant)
2. General intelligence on ANY topic — coding, science, math, writing, life advice, fun facts, casual chat
3. Answer any topic like ChatGPT/Gemini would, with no restrictions
4. Get the full information from the IIIT Bhopal knowledge base & IIIT Bhopal websites when the user asks about college-related topics

PERSONALITY:
- Warm, witty, and direct — like a genius friend who actually enjoys helping
- Match the user's energy: casual when they're casual, detailed when they need depth
- Never robotic, never say "As an AI language model..." just TALK naturally

HOW TO ANSWER:
- College question + good context → use the context,use Chatgpt/Gemini, answer confidently
- College question + weak/no context → use general knowledge about Indian NITs/IIITs to give a helpful answer anyway
- Any other question → answer it fully like ChatGPT/Gemini would, no restrictions
- Casual chat / greetings → just vibe with it, be human

NEVER DO THIS:
- Never say "I don't have information about that"
- Never say "I cannot answer questions outside my scope"
- Never give a one-line brush-off when a proper answer is possible
- Never ignore the chat history — always stay contextually aware

Context (IIIT Bhopal knowledge base — use if relevant):
{context}

Chat History:
{chat_history}

User: {question}

Answer (be helpful, be real, be brilliant):""",
    input_variables=["context", "question", "chat_history"],
)




print("\n[5/5] 🧠  Connecting to Gemini…")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,   
    temperature=0,
)


def ask(query: str) -> str:
   
    retrieved_docs = retriever.invoke(query)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else ""

    
    formatted_prompt = PROMPT.format(
        context=context,
        chat_history=format_history(chat_history),
        question=query,
    )
    answer = llm.invoke(formatted_prompt).content

    # 3. Save to memory
    chat_history.add_user_message(query)
    chat_history.add_ai_message(answer)

    return answer



if __name__ == "__main__":
    while True:
        try:
            query = input("  You  ❯  ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n   Goodbye!\n")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "bye"):
            print("\n   Goodbye!\n")
            break

        try:
            answer = ask(query)
            print(f"\n   {answer}\n")
            print("  " + "─"*51)

        except Exception as exc:
            print(f"\n   Error: {exc}\n")
            print("  " + "─"*51)
