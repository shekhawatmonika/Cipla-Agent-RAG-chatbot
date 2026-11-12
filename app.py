
import json
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Cipla RAG Chatbot",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stChatMessage {
    max-width: 100%;
}
.stMarkdown {
    overflow-wrap: anywhere;
}
</style>
""", unsafe_allow_html=True)

st.title("💊 Cipla RAG Chatbot")

st.write(
    "Ask questions about Cipla, its products, "
    "therapies, careers and healthcare information."
)


# =====================================================
# 2. FIND THE JSON DATASET
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILES = [
    BASE_DIR / "cipla_clean.json",
    BASE_DIR / "data" / "cipla_clean.json",
    BASE_DIR / "cipla_full_site.json",
    BASE_DIR / "data" / "cipla_full_site.json"
]


def find_data_file():

    for file_path in DATA_FILES:

        if file_path.exists():
            return file_path

    return None


# =====================================================
# 3. CLEAN WEBSITE TEXT
# =====================================================

def clean_document_text(text):

    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove CSS fragments
    text = re.sub(
        r"\.cls-\d+\s*\{[^}]*\}",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\.cls-\d+", " ", text)

    # Remove CSS properties
    text = re.sub(
        r"\b(fill|stroke|stroke-width|stroke-linecap|stroke-linejoin)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove common website navigation
    unwanted_phrases = [
        "Skip to main content",
        "Search Search",
        "Manage Preferences",
        "Enter your keywords",
        "Your search yielded no results",
        "Explore the archive",
        "Visitors book"
    ]

    for phrase in unwanted_phrases:

        text = re.sub(
            re.escape(phrase),
            " ",
            text,
            flags=re.IGNORECASE
        )

    # Remove footer content
    text = re.split(
        r"Privacy Policy|Copyright|All rights reserved",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    # Remove repeated separators and punctuation
    text = re.sub(r"[|•·]{2,}", " ", text)
    text = re.sub(r"[-_]{3,}", " ", text)
    text = re.sub(r"[.,;:]{3,}", " ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip(" |•-_,;:")


# =====================================================
# 4. LOAD JSON DOCUMENTS
# =====================================================

@st.cache_data(show_spinner="Loading Cipla dataset...")


def load_documents(file_path):

    with open(file_path, "r", encoding="utf-8") as file:

        raw_data = json.load(file)

    # Support dictionary-based JSON also
    if isinstance(raw_data, dict):

        if isinstance(raw_data.get("documents"), list):
            raw_data = raw_data["documents"]

        elif isinstance(raw_data.get("data"), list):
            raw_data = raw_data["data"]

        else:
            raw_data = [raw_data]

    documents = []
    seen = set()

    for item in raw_data:

        if not isinstance(item, dict):
            continue

        text = clean_document_text(
            item.get("text", "")
        )

        url = item.get(
            "url",
            "Cipla Official Website"
        )

        # Ignore very short pages
        if len(text) < 100:
            continue

        # Remove duplicate documents
        normalized = text.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        documents.append({
            "text": text,
            "url": url
        })

    return documents


# =====================================================
# 5. QUERY CLEANING AND QUERY REWRITING
# =====================================================

def clean_query(query):

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query.strip()


def rewrite_query(query):

    q = clean_query(query)

    lower_q = q.lower()

    expansions = {
        "asthma": "asthma respiratory breathing inhaler",
        "copd": "COPD respiratory lung disease",
        "diabetes": "diabetes therapy treatment",
        "career": "careers jobs recruitment",
        "job": "careers jobs recruitment",
        "product": "products medicines portfolio",
        "quality": "quality manufacturing compliance",
        "history": "Cipla company history milestones",
        "contact": "contact address phone email",
        "investor": "investor financial reports"
    }

    extra_terms = []

    for keyword, expansion in expansions.items():

        if keyword in lower_q:
            extra_terms.append(expansion)

    if extra_terms:
        q = q + " " + " ".join(extra_terms)

    return q


# =====================================================
# 6. BUILD TF-IDF SEARCH INDEX
# =====================================================

@st.cache_resource


def build_index(document_texts):

    if not document_texts:
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        document_texts
    )

    return vectorizer, matrix


# =====================================================
# 7. RETRIEVER AGENT
# =====================================================

def retrieve_documents(
    query,
    vectorizer,
    matrix,
    documents,
    top_k=5
):

    if vectorizer is None or matrix is None:
        return []

    query_vector = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        score = float(scores[index])

        if score <= 0:
            continue

        results.append({
            "score": score,
            "text": documents[index]["text"],
            "url": documents[index]["url"]
        })

    return results


# =====================================================
# 8. CRITIC AGENT
# =====================================================

def critic_agent(results):

    clean_results = []

    unwanted_signals = [
        "your search yielded no results",
        "enter your keywords",
        "skip to main content",
        "explore the archive",
        "visitors book"
    ]

    for result in results:

        text = result["text"].lower()

        # Reject navigation pages
        if any(
            signal in text
            for signal in unwanted_signals
        ):
            continue

        # Reject pages with excessive website noise
        if text.count("manage preferences") > 2:
            continue

        clean_results.append(result)

    return clean_results


# =====================================================
# 9. SPLIT DOCUMENTS INTO SENTENCES
# =====================================================

def split_sentences(text):

    # Split using sentence-ending punctuation
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        # Ignore very short fragments
        if len(sentence) < 35:
            continue

        # Ignore fragments with too few words
        if len(sentence.split()) < 6:
            continue

        # Ignore punctuation-only content
        if not re.search(r"[A-Za-z]", sentence):
            continue

        cleaned_sentences.append(sentence)

    return cleaned_sentences


# =====================================================
# 10. ANSWER GENERATOR
# =====================================================

def generate_answer(
    query,
    results,
    vectorizer,
    role
):

    if not results:

        return (
            "I could not find relevant information "
            "in the Cipla dataset. Please try "
            "rephrasing your question."
        ), []

    sentence_records = []

    # Extract sentences from retrieved documents
    for result in results:

        sentences = split_sentences(
            result["text"]
        )

        for sentence in sentences:

            sentence_records.append({
                "sentence": sentence,
                "url": result["url"],
                "document_score": result["score"]
            })

    if not sentence_records:

        return (
            "I found related pages but could not "
            "extract a clear answer. Please rephrase "
            "your question."
        ), []

    sentences = [
        item["sentence"]
        for item in sentence_records
    ]

    # IMPORTANT:
    # Rank sentences using the original question,
    # not the expanded query.

    query_vector = vectorizer.transform(
        [clean_query(query)]
    )

    sentence_matrix = vectorizer.transform(
        sentences
    )

    sentence_scores = cosine_similarity(
        query_vector,
        sentence_matrix
    ).flatten()

    ranked_sentences = []

    for index, score in enumerate(sentence_scores):

        document_score = sentence_records[index][
            "document_score"
        ]

        combined_score = (
            0.80 * float(score)
            + 0.20 * document_score
        )

        ranked_sentences.append((
            combined_score,
            sentence_records[index]
        ))

    ranked_sentences.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Select useful and unique sentences
    selected = []
    seen = set()
    sources = []

    for score, item in ranked_sentences:

        sentence = item["sentence"]

        normalized = sentence.lower()

        if normalized in seen:
            continue

        # Ignore weak sentence matches
        if score < 0.04:
            continue

        seen.add(normalized)

        selected.append(sentence)

        if item["url"] not in sources:
            sources.append(item["url"])

        if len(selected) >= 5:
            break

    if not selected:

        return (
            "I could not identify a sufficiently "
            "relevant answer in the available data.",
            []
        )

    # Create readable answer
    if role == "Patient":

        heading = (
            "### Simple Explanation"
        )

        max_chars = 1300

    else:

        heading = (
            "### Detailed Information"
        )

        max_chars = 1800

    answer = heading + "\n\n"

    for sentence in selected:

        if role == "Patient" and len(sentence) > 350:

            sentence = (
                sentence[:347].rsplit(" ", 1)[0]
                + "..."
            )

        answer += "- " + sentence + "\n\n"

        if len(answer) >= max_chars:
            break

    # Medical safety message
    if role == "Patient":

        answer += (
            "\n**Medical Disclaimer:** "
            "This chatbot provides general information "
            "from the Cipla website dataset. It cannot "
            "diagnose a disease or recommend medicines "
            "or dosages. Please consult a qualified "
            "healthcare professional."
        )

    else:

        answer += (
            "\n**Clinical Note:** "
            "This response contains retrieved website "
            "information and is not a clinical decision. "
            "Verify medical information against current "
            "approved prescribing information."
        )

    return answer, sources


# =====================================================
# 11. COMPLETE RAG PIPELINE
# =====================================================

def rag_pipeline(
    query,
    role,
    documents,
    vectorizer,
    matrix
):

    # Agent 1: Planner
    rewritten_query = rewrite_query(query)

    # Agent 2: Retriever
    results = retrieve_documents(
        rewritten_query,
        vectorizer,
        matrix,
        documents,
        top_k=8
    )

    # Retry using original question if needed
    if not results:

        results = retrieve_documents(
            clean_query(query),
            vectorizer,
            matrix,
            documents,
            top_k=8
        )

    # Agent 3: Critic
    results = critic_agent(results)

    # Generate final answer
    answer, sources = generate_answer(
        query,
        results,
        vectorizer,
        role
    )

    return answer, sources, results


# =====================================================
# 12. SIDEBAR SETTINGS
# =====================================================

with st.sidebar:

    st.header("Chatbot Settings")

    role = st.radio(
        "Select your role:",
        ["Patient", "Doctor"]
    )

    st.divider()

    st.write("**Data Source:**")
    st.caption("Cipla website JSON dataset")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =====================================================
# 13. LOAD DATASET AND BUILD INDEX
# =====================================================

data_file = find_data_file()

if data_file is None:

    st.error(
        "cipla_clean.json file not found. "
        "Please keep it in the same folder as app.py "
        "or inside a folder named data."
    )

    st.code("""
project_folder/
│
├── app.py
│
└── cipla_clean.json
""")

    st.stop()


try:

    documents = load_documents(
        str(data_file)
    )

except (
    json.JSONDecodeError,
    OSError,
    UnicodeDecodeError
) as error:

    st.error(
        f"Unable to load JSON dataset: {error}"
    )

    st.stop()


if not documents:

    st.error(
        "No usable documents were found "
        "in the JSON dataset."
    )

    st.stop()


document_texts = tuple(
    item["text"]
    for item in documents
)

vectorizer, matrix = build_index(
    document_texts
)


with st.sidebar:

    st.success(
        f"Documents Loaded: {len(documents)}"
    )

    st.caption(
        f"Dataset: {data_file.name}"
    )


# =====================================================
# 14. CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander("View Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"- [{source}]({source})"
                    )


# =====================================================
# 15. SUGGESTED QUESTIONS
# =====================================================

if not st.session_state.messages:

    st.subheader("Try asking")

    questions = [
        "What is Cipla?",
        "Tell me about Cipla careers.",
        "What are Cipla's main products?",
        "What information does Cipla provide about asthma?"
    ]

    col1, col2 = st.columns(2)

    for index, question in enumerate(questions):

        with [col1, col2][index % 2]:

            if st.button(
                question,
                use_container_width=True
            ):

                st.session_state.pending_question = question

                st.rerun()


# =====================================================
# 16. USER QUESTION AND FINAL RESPONSE
# =====================================================

prompt = st.chat_input(
    "Ask anything about Cipla..."
)

if (
    not prompt
    and st.session_state.get("pending_question")
):

    prompt = st.session_state.pop(
        "pending_question"
    )


if prompt:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    # Generate chatbot response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching Cipla information..."
        ):

            answer, sources, results = rag_pipeline(
                prompt,
                role,
                documents,
                vectorizer,
                matrix
            )

        st.markdown(answer)

        # Display source links
        if sources:

            with st.expander("View Sources"):

                for source in sources:

                    st.markdown(
                        f"- [{source}]({source})"
                    )

        # Optional retrieval details
        with st.expander("Retrieval Details"):

            for index, result in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"Result {index} | "
                    f"Similarity: {result['score']:.3f}"
                )

                st.caption(result["url"])

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
