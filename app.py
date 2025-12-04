import re
import textwrap
from collections import Counter

import requests
import streamlit as st

import json
from openai import OpenAI


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

# using openai for text processing
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")  # or os.getenv() for backup
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


openai_client = get_openai_client()


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


def gpt_similarity_score(
    user_description: str,
    opinion_text: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
):
    """
    Ask GPT to score how similar an opinion is to the user's case (0–5)
    and give a short natural-language reason.

    Returns (score: float or None, reason: str).
    """
    if client is None:
        return None, "GPT client is not configured."

    # keep prompt size reasonable
    snippet = opinion_text[:2000]

    system_prompt = (
        "You are a legal research assistant. "
        "Given a user's case description and a court opinion, you rate how relevant "
        "the opinion is to the user's case on a 0 to 5 scale and explain why. "
        "0 = completely unrelated. 5 = very similar facts and legal issues. "
        "Respond ONLY in valid JSON with keys 'score' (number) and 'reason' (string)."
    )

    user_prompt = f"""
User's case description:
\"\"\"{user_description}\"\"\"

Court opinion excerpt:
\"\"\"{snippet}\"\"\"

Please output JSON like:
{{
  "score": 0-5 number,
  "reason": "short explanation"
}}
"""
try: 
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Extract text
    try:
        content = resp.output[0].content[0].text
        except Exception:
            content = getattr(resp, "output_text", str(resp))

        data = json.loads(content)
        return float(data.get("score", 0)), data.get("reason", "")

except Exception as e:
        msg = str(e)

        if "insufficient_quota" in msg:
            return None, "GPT is unavailable (insufficient OpenAI quota). Using classical NLP only."

        if "429" in msg:
            return None, "GPT temporarily rate-limited. Using classical NLP only."

        return None, f"GPT error: {msg[:120]}"

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
        # --- ID handling (as we fixed before) ---
        opinion_id = item.get("id")
        abs_url = item.get("absolute_url") or ""
        if opinion_id is None and abs_url:
            m = re.search(r"/opinion/(\d+)/", abs_url)
            if m:
                opinion_id = int(m.group(1))
        if opinion_id is None:
            # Skip if we really cannot determine the ID
            continue

        # --- CASE NAME HANDLING ---
        case_name = item.get("case_name") or item.get("caseName")

        # If still missing, try cluster/docket objects
        if not case_name:
            cluster = item.get("cluster") or {}
            if isinstance(cluster, dict):
                case_name = cluster.get("case_name") or cluster.get("caseName")

        if not case_name:
            docket = item.get("docket") or {}
            if isinstance(docket, dict):
                case_name = docket.get("case_name") or docket.get("caseName")

        # If still missing, derive from URL slug
        if not case_name and abs_url:
            m = re.search(r"/opinion/\d+/(.*?)/?$", abs_url)
            if m:
                slug = m.group(1)
                # Make it nicer: replace dashes with spaces, title-case it
                case_name = slug.replace("-", " ").title()
                # Optional: fix " V " to " v. "
                case_name = case_name.replace(" V ", " v. ")

        if not case_name:
            case_name = "Unknown case name"

        # --- Citation ---
        cites = item.get("citation") or item.get("citations")
        if isinstance(cites, list):
            citation = cites[0] if cites else "No citation"
        else:
            citation = cites or "No citation"

        # --- Court ---
        court_field = item.get("court")
        if isinstance(court_field, dict):
            court_name = court_field.get("name") or court_field.get("court_name")
        else:
            court_name = court_field
        court_name = court_name or item.get("court_name") or "Unknown court"

        # --- Date ---
        date = item.get("date_filed") or item.get("dateFiled") or item.get("date") or "Unknown date"

        # --- Web URL for user ---
        if abs_url:
            if abs_url.startswith("http"):
                web_url = abs_url
            else:
                web_url = "https://www.courtlistener.com" + abs_url
        else:
            web_url = f"https://www.courtlistener.com/opinion/{opinion_id}/"

        results.append(
            {
                "id": opinion_id,
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
        return html

    xml = data.get("xml_harvard")
    if xml:
        return strip_xml_tags(xml)

    return "No opinion text available."


# STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config(page_title="Legal Case Finder", layout="wide")

    st.title("🔎 Legal Case Finder (CourtListener)")
    st.write(
        "Describe a legal scenario, and this app retrieves **similar cases** using the "
        "CourtListener API v4 and light NLP for keyword extraction, similarity scoring, "
        "and naive summaries (with optional GPT scoring)."
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

    gpt_model = "gpt-4o-mini"   # cheapest model

    use_gpt_scoring = st.sidebar.checkbox(
        "Use GPT similarity judge (top 5 only)",
        value=False,
        help="Adds semantic relevance scoring when GPT quota is available."
    )

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

        # 1) Call CourtListener
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

        # 🔹 Hard cap by slider, in case API returns more
        base_results = base_results[:num_results]

        # 2) Show extracted keywords
        user_keywords = extract_keywords(user_desc, top_n=12)
        st.markdown("### Extracted Keywords from Your Description")
        if user_keywords:
            st.write(", ".join(user_keywords))
        else:
            st.write("_No significant keywords detected (input may be too short)._")

        # 3) First pass: build enriched_results with ONLY Jaccard
        enriched_results = []

        for result in base_results:
            opinion_id = result["id"]

            try:
                text = get_opinion_text(token, opinion_id)
            except Exception as e:
                text = f"Error retrieving opinion text: {e}"

            summary = summarize_text(text, max_sentences=3)

            # Cheap keyword-based similarity (Jaccard)
            case_tokens = tokenize(text)
            cheap_score = jaccard_score(user_keywords, case_tokens) if use_rerank else None

            enriched = {**result}
            enriched["text"] = text
            enriched["summary"] = summary
            enriched["similarity"] = cheap_score      # Jaccard
            enriched["gpt_score"] = None              # placeholder
            enriched["gpt_reason"] = ""               # placeholder
            enriched_results.append(enriched)

        # 4) Sort ALL results by Jaccard first (baseline ranking)
        if use_rerank:
            enriched_results.sort(
                key=lambda x: (x["similarity"] if x["similarity"] is not None else 0.0),
                reverse=True,
            )

        # 5) GPT only reranks the TOP N according to Jaccard
        max_gpt = 5  # number of top-Jaccard cases to refine with GPT

        if use_gpt_scoring:
            if openai_client is None:
                st.warning("GPT scoring enabled, but OPENAI_API_KEY is not configured in secrets.")
            else:
                top_for_gpt = enriched_results[:max_gpt]
                rest = enriched_results[max_gpt:]

                # call GPT on these top cases
                for case in top_for_gpt:
                    gpt_score, gpt_reason = gpt_similarity_score(
                        user_description=user_desc,
                        opinion_text=case["text"],
                        client=openai_client,
                        model=gpt_model,
                    )
                    time.sleep(0.7) #prevents rate limit
                    case["gpt_score"] = gpt_score
                    case["gpt_reason"] = gpt_reason

                # sort ONLY the top segment by GPT score, keep rest in Jaccard order
                top_for_gpt.sort(
                    key=lambda x: (x["gpt_score"] if x["gpt_score"] is not None else -1),
                    reverse=True,
                )

                enriched_results = top_for_gpt + rest

        # 6) Display results
        st.markdown("## Results")

        for i, case in enumerate(enriched_results, start=1):
            similarity = case.get("similarity")
            sim_str = f"{similarity:.3f}" if similarity is not None else "N/A"

            gpt_score = case.get("gpt_score")
            gpt_reason = case.get("gpt_reason") or ""

            with st.expander(f"{i}. {case['case_name']}"):
                cols = st.columns([3, 2])
                with cols[0]:
                    st.markdown(f"**Citation:** {case['citation']}")
                    st.markdown(f"**Court:** {case['court']}")
                    st.markdown(f"**Date:** {case['date']}")
                    st.markdown(f"**Relevance score (Jaccard):** {sim_str}")
                    if gpt_score is not None:
                        st.markdown(f"**Relevance score (GPT, 0–5):** {gpt_score:.2f}")
                with cols[1]:
                    st.markdown(
                        f"[Open full case on CourtListener]({case['web_url']})"
                    )

                if gpt_reason:
                    st.markdown("**GPT explanation of relevance:**")
                    st.write(gpt_reason)

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
