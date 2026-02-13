import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import os
import hashlib
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict

# =========================
# CONFIG
# =========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "meta-llama/llama-3.1-8b-instruct"


# =========================
# STORAGE
# =========================

class InMemoryStorage:

    def __init__(self):
        self.leads = []
        self.chatbots = {}

    def save_lead(self, chatbot_id, company_name, name, email, phone, session_id, questions, conversation):

        lead = {
            "chatbot_id": chatbot_id,
            "company_name": company_name,
            "name": name,
            "email": email,
            "phone": phone,
            "session_id": session_id,
            "questions": questions,
            "conversation": conversation,
            "created": datetime.now()
        }

        self.leads.append(lead)

    def get_leads(self):
        return self.leads


storage = InMemoryStorage()


# =========================
# SCRAPER
# =========================

class FastScraper:

    def __init__(self):

        self.headers = {
            "User-Agent": "Mozilla/5.0"
        }

        self.timeout = 8

    def scrape_page(self, url):

        try:

            r = requests.get(url, headers=self.headers, timeout=self.timeout)

            if r.status_code != 200:
                return None

            soup = BeautifulSoup(r.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            text = soup.get_text(separator="\n")

            lines = [
                line.strip()
                for line in text.split("\n")
                if len(line.strip()) > 30
            ]

            content = "\n".join(lines)

            return {
                "url": url,
                "content": content[:8000]
            }

        except:
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

            futures = [
                executor.submit(self.scrape_page, url)
                for url in urls
            ]

            for f in as_completed(futures):

                result = f.result()

                if result:
                    pages.append(result)

        return pages


# =========================
# AI
# =========================

class SmartAI:

    def __init__(self):
        self.cache = {}

    def call(self, prompt):

        if not OPENROUTER_API_KEY:
            return "API key missing"

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
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 500
            }

        )

        if response.status_code != 200:
            return "AI Error"

        answer = response.json()["choices"][0]["message"]["content"]

        self.cache[cache_key] = answer

        return answer


# =========================
# CHATBOT
# =========================

class UniversalChatbot:

    def __init__(self, company, url):

        self.company = company
        self.url = url

        self.pages = []

        self.chunks = []

        self.ai = SmartAI()

        self.ready = False

    # =========================
    # INIT
    # =========================

    def initialize(self):

        scraper = FastScraper()

        self.pages = scraper.scrape_website(self.url)

        self.create_chunks()

        self.ready = True


    # =========================
    # CREATE CHUNKS
    # =========================

    def create_chunks(self):

        chunks = []

        for page in self.pages:

            text = page["content"]

            for i in range(0, len(text), 500):

                chunk = text[i:i+500]

                chunks.append(chunk)

        self.chunks = chunks


    # =========================
    # FIND RELEVANT CONTEXT
    # =========================

    def get_relevant_chunks(self, question):

        keywords = question.lower().split()

        scored = []

        for chunk in self.chunks:

            score = 0

            for word in keywords:

                if word in chunk.lower():
                    score += 1

            if score > 0:
                scored.append((score, chunk))

        scored.sort(reverse=True)

        best = scored[:5]

        context = "\n\n".join([c[1] for c in best])

        return context


    # =========================
    # ASK
    # =========================

    def ask(self, question):

        if not self.ready:
            return "Bot not ready"

        context = self.get_relevant_chunks(question)

        if not context:
            return "I couldn't find that information on our website."

        prompt = f"""

You are an official chatbot for {self.company}.

CRITICAL RULES:

Answer ONLY using the website context below.

DO NOT make up information.

DO NOT guess.

If answer not present, say:
"I couldn't find that information on our website."

WEBSITE CONTEXT:
{context}


QUESTION:
{question}


ANSWER:

"""

        answer = self.ai.call(prompt)

        return answer


# =========================
# STREAMLIT UI
# =========================

def main():

    st.title("AI Website Chatbot")

    if "bot" not in st.session_state:
        st.session_state.bot = None

    company = st.text_input("Company Name")

    url = st.text_input("Website URL")

    if st.button("Create Bot"):

        bot = UniversalChatbot(company, url)

        with st.spinner("Scraping website..."):
            bot.initialize()

        st.session_state.bot = bot

        st.success("Bot ready")


    if st.session_state.bot:

        question = st.chat_input("Ask question")

        if question:

            with st.spinner("Thinking..."):

                answer = st.session_state.bot.ask(question)

            st.write(answer)


# =========================

if __name__ == "__main__":
    main()
