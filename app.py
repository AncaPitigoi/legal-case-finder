import re
import textwrap
from collections import Counter

import requests
import streamlit as st

# -----------------------------
# API & CONFIG
# -----------------------------

BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# Minimal English stopword list for simple NLP
STOPWORDS = {
    "the", "and", "or", "a", "an", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those", "he", "she", "they", "them",
    "his", "her", "their", "we", "you", "your", "i", "my", "me", "our",
    "but", "if", "so", "not", "no", "can", "could", "would", "should",
    "about", "into", "over", "under", "between", "because", "up", "down"
}


def get_api_token():
    """
    Get API token from Streamlit secrets or sidebar input.
    If COURTLISTENER_API_TOKEN exists in st.secrets, use that.
    Otherwise, show a password field in the sidebar.
    """
    token = st.secrets.get("COURTLISTENER_API_TOKEN", None)
    if token:
        # Debug helper – shows only first few chars so you know it's loaded.
        st.sidebar.write(f"Loaded token starting with: {token[:4]}***")
        return token

    token = st.sidebar.text_input(
        "CourtListener API Token",
        type="password",
        help="Paste your CourtListener API token here if not using st.secrets."
    )
    return token.strip() if token else None


def make_headers(token: str):
    return {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
    }


# -----------------------------
# SIMPLE NLP UTILITIES
# -----------------------------

def tokenize(text: str):
    """Lowercase, remove non-letters, split into tokens."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def extract_keywords(text: str, top_n: int = 10):
    """Naive keyword extraction using frequency, ignoring stopwords."""
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    if not tokens:
        return []

    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(top_n)]


def jaccard_score(a_tokens, b_tokens):
    """Jaccard similarity between two token sets."""
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a or not set_b:
        return 0.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union)


def summarize_text(text: str, max_sentences: int = 3):
    """Very simple extractive summary: first few sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    summary = " ".join(sentences[:max_sentences])
    return summary


def strip_xml_tags(xml: str) -> str:
    """
    Very rough stripper for xml_harvard content.
    Just removes <...> tags and compresses whitespace.
    """
    no_tags = re.sub(r"<[^>]+>", " ", xml)
    no_tags = re.sub(r"\s+", " ", no_tags)
    return no_tags.strip()


# -----------------------------
# COURTLISTENER HELPERS (v4)
# -----------------------------

def search_cases(token: str, query: str, page_size: int = 5, jurisdiction: str | None = None):
    params = {
        "q": query,
        "type": "o",          # opinions
        "page_size": page_size,
    }
    if jurisdiction and jurisdiction != "all":
        params["jurisdiction"] = jurisdiction

    r = requests.get(f"{BASE_URL}/search/", headers=make_headers(token), params=params)
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data.get("results", []):
        # 1) Try the id field first
        opinion_id = item.get("id")

        # 2) If it's missing/None, try to extract from absolute_url
        if opinion_id is None:
            abs_url = item.get("absolute_url") or ""
            m = re.search(r"/opinion/(\d+)/", abs_url)
            if m:
                opinion_id = int(m.group(1))

        # 3) If we *still* don't have an id, just skip this result
        if opinion_id is None:
            continue

        case_name = item.get("case_name") or "Unknown case name"

        cites = item.get("citation") or item.get("citations")
        if isinstance(cites, list):
            citation = cites[0] if cites else "No citation"
        else:
            citation = cites or "No citation"

        court_field = item.get("court")
        if isinstance(court_field, dict):
            court_name = court_field.get("name") or court_field.get("court_name")
        else:
            court_name = court_field
        court_name = court_name or item.get("court_name") or "Unknown court"

        date = item.get("date_filed") or item.get("dateFiled") or item.get("date") or "Unknown date"

        abs_url = item.get("absolute_url") or f"/opinion/{opinion_id}/"
        if abs_url.startswith("http"):
            web_url = abs_url
        else:
            web_url = "https://www.courtlistener.com" + abs_url

        results.append(
            {
                "id": opinion_id,              # now guaranteed to be an int
                "case_name": case_name,
                "citation": citation,
                "court": court_name,
                "date": date,
                "web_url": web_url,
            }
        )

    return results


def get_opinion_text(token: str, opinion_id: int):
    """
    Get full text of an opinion via /opinions/{id}/ in API v4.
    Tries plain_text, then html, then xml_harvard.
    """
    if opinion_id is None:
    return "No opinion ID available for this result."
    url = f"{BASE_URL}/opinions/{opinion_id}/"
    r = requests.get(url, headers=make_headers(token))
    r.raise_for_status()
    data = r.json()

    text = data.get("plain_text")
    if text:
        return text

    html = data.get("html")
    if html:
        # You could keep HTML, but for now just return it verbatim.
        return html

    xml = data.get("xml_harvard")
    if xml:
        return strip_xml_tags(xml)

    return "No opinion text available."


# -----------------------------
# STREAMLIT APP
# -----------------------------

def main():
    st.set_page_config(page_title="Legal Case Finder", layout="wide")

    st.title("🔎 Legal Case Finder (CourtListener)")
    st.write(
        "Describe a legal scenario, and this app retrieves **similar cases** using the "
        "CourtListener API v4 and light NLP for keyword extraction, similarity scoring, "
        "and naive summaries."
    )

    st.sidebar.header("Settings")

    token = get_api_token()
    if not token:
        st.sidebar.warning("Enter your CourtListener API token to begin.")
        st.stop()

    jurisdiction_label = st.sidebar.selectbox(
        "Jurisdiction filter",
        options=["All", "New York (ny)", "California (ca)", "Federal (us)"],
        index=0,
        help="Optional: Restrict to a jurisdiction. Leave as 'All' to search broadly."
    )
    jurisdiction_map = {
        "All": "all",
        "New York (ny)": "ny",
        "California (ca)": "ca",
        "Federal (us)": "us",
    }
    jurisdiction = jurisdiction_map[jurisdiction_label]

    num_results = st.sidebar.slider(
        "Number of results",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
    )

    use_rerank = st.sidebar.checkbox(
        "Use NLP re-ranking (keyword overlap)",
        value=True,
        help="Rerank API results using keyword overlap between your description and each opinion."
    )

    st.subheader("Describe Your Case")
    user_desc = st.text_area(
        "Facts, issues, and context:",
        height=150,
        placeholder="Example: My client slipped on ice outside a supermarket, "
                    "store knew about the hazard, premises liability, New York..."
    )

    if st.button("Search similar cases"):
        if not user_desc.strip():
            st.warning("Please enter a description first.")
            st.stop()

        try:
            with st.spinner("Searching CourtListener (v4)..."):
                base_results = search_cases(
                    token=token,
                    query=user_desc,
                    page_size=num_results,
                    jurisdiction=jurisdiction,
                )
        except Exception as e:
            st.error(f"Error while searching CourtListener: {e}")
            st.stop()

        if not base_results:
            st.info("No matching cases found.")
            st.stop()

        # NLP: extract keywords from user description
        user_keywords = extract_keywords(user_desc, top_n=12)
        st.markdown("### Extracted Keywords from Your Description")
        if user_keywords:
            st.write(", ".join(user_keywords))
        else:
            st.write("_No significant keywords detected (input may be too short)._")

        enriched_results = []
        if use_rerank:
            st.info(
                "Re-ranking results using keyword overlap. "
                "This may take a few extra seconds."
            )

        for result in base_results:
            opinion_id = result["id"]
            try:
                text = get_opinion_text(token, opinion_id)
            except Exception as e:
                text = f"Error retrieving opinion text: {e}"

            summary = summarize_text(text, max_sentences=3)

            if use_rerank:
                case_tokens = tokenize(text)
                score = jaccard_score(user_keywords, case_tokens)
            else:
                score = None

            enriched = {**result}
            enriched["text"] = text
            enriched["summary"] = summary
            enriched["similarity"] = score
            enriched_results.append(enriched)

        # Optional: sort by similarity
        if use_rerank:
            enriched_results.sort(
                key=lambda x: (x["similarity"] if x["similarity"] is not None else 0.0),
                reverse=True,
            )

        st.markdown("## Results")

        for i, case in enumerate(enriched_results, start=1):
            similarity = case.get("similarity")
            sim_str = f"{similarity:.3f}" if similarity is not None else "N/A"

            with st.expander(f"{i}. {case['case_name']}"):
                cols = st.columns([3, 2])
                with cols[0]:
                    st.markdown(f"**Citation:** {case['citation']}")
                    st.markdown(f"**Court:** {case['court']}")
                    st.markdown(f"**Date:** {case['date']}")
                    st.markdown(f"**Relevance score (NLP):** {sim_str}")
                with cols[1]:
                    st.markdown(
                        f"[Open full case on CourtListener]({case['web_url']})"
                    )

                st.markdown("**Summary (naive, first few sentences):**")
                st.write(textwrap.fill(case["summary"], width=90))

                show_full = st.checkbox(
                    f"Show opinion excerpt for result {i}",
                    key=f"show_full_{i}",
                )
                if show_full:
                    st.markdown("**Opinion Text (excerpt):**")
                    st.write(textwrap.fill(case["text"][:3000], width=90))
                    st.write("… [truncated] …")


if __name__ == "__main__":
    main()
