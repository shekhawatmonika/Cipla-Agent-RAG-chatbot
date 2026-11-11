import json
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Cipla RAG Chatbot",
    page_icon="💊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 1rem;
        }
        .answer-note {
            color: #666;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💊 Cipla RAG Chatbot")
st.caption(
    "Ask questions about Cipla, its company profile, careers, therapies, "
    "products, manufacturing, quality, sustainability and other topics "
    "covered by the supplied website data."
)

# ============================================================
# 2. USER MODE
# ============================================================
if "role" not in st.session_state:
    st.session_state.role = "Patient"

mode = st.radio(
    "Choose your answer style",
    options=["Patient", "Doctor"],
    horizontal=True,
    index=0 if st.session_state.role == "Patient" else 1,
)
st.session_state.role = mode

if mode == "Doctor":
    st.info(
        "Doctor mode: responses use the more detailed wording available in "
        "the source."
    )
else:
    st.info(
        "Patient mode: responses use shorter, simpler wording where possible."
    )

st.divider()

# ============================================================
# 3. LOAD AND CLEAN THE JSON DATA
# ============================================================


def find_data_file():
    """Find cipla_clean.json in common project locations."""
    possible_paths = [
        Path("data/cipla_clean.json"),
        Path("cipla_clean.json"),
        Path(__file__).resolve().parent / "data" / "cipla_clean.json",
        Path(__file__).resolve().parent / "cipla_clean.json",
    ]

    for path in possible_paths:
        if path.exists() and path.is_file():
            return path

    return None


def clean_text(text):
    """Remove crawler noise and normalize whitespace."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<script.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"\.cls-\d+\s*\{[^}]*\}", " ", text, flags=re.I)
    text = re.sub(r"\.cls-\d+", " ", text, flags=re.I)

    noise_patterns = [
        r"Skip to main content",
        r"Search Search",
        r"Enter your keywords",
        r"Your search yielded no results",
        r"Manage Preferences",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text, flags=re.I)

    # Remove common footer material only when it appears near the end.
    text = re.sub(r"\s+Privacy Policy.*$", "", text, flags=re.I | re.S)
    text = re.sub(r"\s+Copyright.*$", "", text, flags=re.I | re.S)

    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n-|")


@st.cache_data(show_spinner="Loading Cipla website data...")
def load_documents(file_path_string):
    """Read JSON records and return usable page documents."""
    file_path = Path(file_path_string)

    with file_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list):
        raise ValueError("The JSON file must contain a list of page records.")

    documents = []
    seen = set()

    for item in raw_data:
        if not isinstance(item, dict):
            continue

        text = clean_text(item.get("text", ""))
        url = str(item.get("url", "Cipla official website")).strip()

        # Ignore very short pages and repeated crawler/menu pages.
        if len(text) < 100:
            continue

        fingerprint = re.sub(r"\W+", "", text.lower())[:500]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        documents.append(
            {
                "text": text,
                "url": url or "Cipla official website",
            }
        )

    return documents


data_file = find_data_file()

if data_file is None:
    st.error(
        "Could not find `cipla_clean.json`. Keep the JSON file in the same "
        "folder as `app.py`, or place it inside a folder named `data`."
    )
    st.stop()

try:
    documents = load_documents(str(data_file))
except (OSError, json.JSONDecodeError, ValueError) as error:
    st.error(f"Unable to load the JSON data: {error}")
    st.stop()

if not documents:
    st.error("No usable website content was found in cipla_clean.json.")
    st.stop()

# ============================================================
# 4. PREPARE THE TF-IDF SEARCH INDEX
# ============================================================


@st.cache_resource
def create_search_index(document_texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    document_matrix = vectorizer.fit_transform(document_texts)
    return vectorizer, document_matrix


document_texts = [item["text"] for item in documents]

try:
    vectorizer, document_matrix = create_search_index(document_texts)
except ValueError:
    st.error("The supplied JSON does not contain enough searchable text.")
    st.stop()

# ============================================================
# 5. QUERY REWRITING
# ============================================================


def rewrite_query(query):
    """Add useful search terms for common Cipla topics."""
    original = query.strip()
    lower_query = original.lower()

    expansions = {
        "cipla": "Cipla company about us pharmaceutical history purpose",
        "asthma": "asthma respiratory inhaler inhalation therapy",
        "copd": "COPD chronic obstructive pulmonary disease respiratory therapy",
        "career": "careers jobs recruitment work life at Cipla",
        "job": "careers jobs recruitment work life at Cipla",
        "product": "products medicine molecule therapeutic categories",
        "manufacturing": "manufacturing facilities API formulation production",
        "quality": "quality assurance quality control regulatory standards",
        "sustainability": "sustainability ESG environment community governance",
        "investor": "investor information annual report financial results",
        "history": "Cipla history founded 1935 Dr K A Hamied",
        "contact": "contact address email phone registered office",
    }

    additions = [
        extra for keyword, extra in expansions.items() if keyword in lower_query
    ]

    if additions:
        return original + " " + " ".join(additions)

    return original


# ============================================================
# 6. RETRIEVAL
# ============================================================


def retrieve_documents(query, top_k=4):
    """Return the most relevant pages for the user's question."""
    search_query = rewrite_query(query)
    query_vector = vectorizer.transform([search_query])
    scores = cosine_similarity(query_vector, document_matrix).flatten()

    ranked_indices = scores.argsort()[::-1]

    results = []
    for index in ranked_indices:
        score = float(scores[index])

        if score <= 0:
            break

        results.append(
            {
                "score": score,
                "text": documents[index]["text"],
                "url": documents[index]["url"],
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# 7. BUILD A CLEAN, SOURCE-BASED ANSWER
# ============================================================


def split_into_sentences(text):
    """Split page text into readable sentence-like units."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) > 35]


def select_relevant_sentences(query, retrieved, max_sentences=5):
    """Select relevant sentences using the original user query for relevance."""
    query_terms = {
        term.lower() for term in re.findall(r"[a-zA-Z0-9]+", query) if len(term) > 2
    }

    candidates = []
    seen_sentences = set()

    for page in retrieved:
        for sentence in split_into_sentences(page["text"]):
            # Clean up unwanted commas, bullets, and numbering artifacts at the start/within sentences
            cleaned_sentence = re.sub(r"^[-•*\d,.\s]+", "", sentence).strip()
            if not cleaned_sentence:
                continue

            normalized = re.sub(r"\W+", "", cleaned_sentence.lower())

            if len(normalized) < 35 or normalized in seen_sentences:
                continue

            seen_sentences.add(normalized)

            sentence_terms = {
                term.lower()
                for term in re.findall(r"[a-zA-Z0-9]+", cleaned_sentence)
                if len(term) > 2
            }

            overlap = len(query_terms.intersection(sentence_terms))
            relevance = overlap + (page["score"] * 2)

            if overlap > 0 or page["score"] >= 0.12:
                candidates.append(
                    {
                        "sentence": cleaned_sentence,
                        "score": relevance,
                        "url": page["url"],
                    }
                )

    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    for candidate in candidates:
        if len(selected) >= max_sentences:
            break
        selected.append(candidate)

    return selected


def format_answer(query, retrieved, role):
    """Create a clean bulleted answer with source links using the original question."""
    if not retrieved or retrieved[0]["score"] < 0.06:
        return (
            "I could not find a sufficiently relevant answer in the supplied "
            "Cipla website data. Try asking with a more specific topic, such "
            "as Cipla history, careers, manufacturing, quality or sustainability."
        ), []

    max_sentences = 8 if role == "Doctor" else 5
    selected = select_relevant_sentences(
        query,
        retrieved,
        max_sentences=max_sentences,
    )

    if not selected:
        best_page = retrieved[0]
        fallback_sentence = re.sub(r"^[-•*\d,.\s]+", "", best_page["text"][:700]).strip()
        selected = [
            {
                "sentence": fallback_sentence,
                "url": best_page["url"],
                "score": best_page["score"],
            }
        ]

    answer_lines = []
    sources = []
    for item in selected:
        answer_lines.append(f"- {item['sentence']}")
        if item["url"] not in sources:
            sources.append(item["url"])

    answer = "\n".join(answer_lines)

    if role == "Patient":
        answer += (
            "\n\n**Important:** This chatbot retrieves information from the "
            "provided Cipla website dataset. It does not diagnose conditions "
            "or recommend medicines. Please consult a qualified healthcare "
            "professional for personal medical advice."
        )

    return answer, sources


def get_answer(query, role):
    retrieved = retrieve_documents(query, top_k=5)
    answer, sources = format_answer(query, retrieved, role)
    return answer, sources, retrieved


# ============================================================
# 8. SIDEBAR INFORMATION
# ============================================================
with st.sidebar:
    st.header("Chatbot information")
    st.success(f"Documents loaded: {len(documents)}")
    st.caption(f"Data file: {data_file.name}")
    st.caption("Search method: TF-IDF + cosine similarity")
    st.caption("Answer method: Extractive retrieval from supplied pages")
    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Suggested questions**")
    suggestions = [
        "What is Cipla?",
        "When was Cipla incorporated?",
        "Tell me about Cipla careers.",
        "What are Cipla's core values?",
        "Explain Cipla's sustainability initiatives.",
        "What does Cipla say about product quality?",
    ]

    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion

# ============================================================
# 9. CHAT HISTORY
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.markdown(f"- [{source}]({source})")

# ============================================================
# 10. CHAT INPUT AND RESPONSE
# ============================================================
pending_question = st.session_state.pop("pending_question", None)
typed_question = st.chat_input("Ask a question about Cipla...")
question = pending_question or typed_question

if question:
    question = question.strip()

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching Cipla website data..."):
                answer, sources, retrieved = get_answer(
                    question,
                    st.session_state.role,
                )

            st.markdown(answer)

            if sources:
                with st.expander("View sources"):
                    for source in sources:
                        st.markdown(f"- [{source}]({source})")

            with st.expander("Search details"):
                for number, result in enumerate(retrieved, start=1):
                    st.write(
                        f"{number}. Similarity score: {result['score']:.3f}"
                    )
                    st.caption(result["url"])

        assistant_message = answer

        if sources:
            assistant_message += "\n\n**Sources:**\n" + "\n".join(
                f"- {source}" for source in sources
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_message,
                "sources": sources,
            }
        )