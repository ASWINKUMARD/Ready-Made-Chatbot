import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================
# CONFIG
# ==========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.1-8b-instruct"

# ==========================
# SCRAPER
# ==========================

class WebsiteScraper:

    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.timeout = 10

    def scrape_page(self, url):
        try:
            r = requests.get(url, headers=self.headers, timeout=self.timeout)
            if r.status_code != 200:
                return None

            soup = BeautifulSoup(r.text, "html.parser")

            # Remove only script and style
            for tag in soup(["script", "style"]):
                tag.decompose()

            main = soup.find("main")

            if main:
                text = main.get_text(separator="\n")
            else:
                text = soup.get_text(separator="\n")

            lines = [
                line.strip()
                for line in text.split("\n")
                if len(line.strip()) > 25
            ]

            content = "\n".join(lines)

            return content[:15000]

        except Exception as e:
            print("Scrape error:", e)
            return None

    def scrape_website(self, base_url):

        if not base_url.startswith("http"):
            base_url = "https://" + base_url

        urls = [
            base_url,
            base_url + "/about",
            base_url + "/services",
            base_url + "/products",
            base_url + "/contact"
        ]

        pages = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.scrape_page, u) for u in urls]

            for f in as_completed(futures):
                result = f.result()
                if result:
                    pages.append(result)

        return pages


# ==========================
# AI
# ==========================

class LLM:

    def __init__(self):
        self.cache = {}

    def generate(self, prompt):

        if not OPENROUTER_API_KEY:
            return "⚠️ API key not set"

        cache_key = hashlib.md5(prompt.encode()).hexdigest()

        if cache_key in self.cache:
            return self.cache[cache_key]

        response = requests.post(
            OPENROUTER_API_BASE,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500
            }
        )

        if response.status_code != 200:
            print(response.text)
            return "⚠️ AI error"

        answer = response.json()["choices"][0]["message"]["content"]

        self.cache[cache_key] = answer

        return answer


# ==========================
# CHATBOT (RAG)
# ==========================

class WebsiteChatbot:

    def __init__(self, company, url):

        self.company = company
        self.url = url
        self.pages = []
        self.chunks = []
        self.ai = LLM()
        self.ready = False

    # ======================
    # Initialize
    # ======================

    def initialize(self):

        scraper = WebsiteScraper()
        self.pages = scraper.scrape_website(self.url)

        self.create_chunks()

        self.ready = True

    # ======================
    # Chunking
    # ======================

    def create_chunks(self):

        chunks = []

        for page in self.pages:

            for i in range(0, len(page), 600):
                chunk = page[i:i+600]
                chunks.append(chunk)

        self.chunks = chunks

    # ======================
    # Retrieval
    # ======================

    def retrieve(self, question):

        question_lower = question.lower()

        scored = []

        for chunk in self.chunks:

            chunk_lower = chunk.lower()
            score = 0

            if question_lower in chunk_lower:
                score += 10

            for word in question_lower.split():
                if len(word) > 3 and word in chunk_lower:
                    score += 2

            if score > 0:
                scored.append((score, chunk))

        scored.sort(reverse=True)

        best = scored[:5]

        return "\n\n".join([c[1] for c in best])

    # ======================
    # Ask
    # ======================

    def ask(self, question):

        if not self.ready:
            return "Bot not ready"

        context = self.retrieve(question)

        # fallback if nothing found
        if not context:
            context = "\n\n".join(self.chunks[:8])

        prompt = f"""
You are the official AI assistant for {self.company}.

STRICT RULES:
- Answer ONLY using the website context below.
- Do NOT guess.
- Do NOT use outside knowledge.
- If answer not found, say:
  "I couldn't find that information on our website."

WEBSITE CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        return self.ai.generate(prompt)


# ==========================
# STREAMLIT UI
# ==========================

def main():

    st.set_page_config(page_title="Website AI Chatbot", layout="wide")

    st.title("🚀 Website AI Chatbot (RAG Version)")

    if "bot" not in st.session_state:
        st.session_state.bot = None

    company = st.text_input("Company Name")
    url = st.text_input("Website URL")

    if st.button("Create Chatbot"):

        if not company or not url:
            st.warning("Enter company name and URL")
            return

        bot = WebsiteChatbot(company, url)

        with st.spinner("Scraping website..."):
            bot.initialize()

        st.session_state.bot = bot

        st.success("✅ Chatbot Ready!")

    if st.session_state.bot:

        question = st.chat_input("Ask about the website...")

        if question:

            with st.spinner("Thinking..."):
                answer = st.session_state.bot.ask(question)

            st.write(answer)


if __name__ == "__main__":
    main()
