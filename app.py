import re
import textwrap
from collections import Counter

import requests
import streamlit as st

import json
from openai import OpenAI
import time


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
    """
    Retrieve an OpenAI client using:
    1. OPENAI_API_KEY from Streamlit secrets, if available.
    2. Otherwise, a user-provided API key via the sidebar.
    """

    # 1. Try Streamlit secrets
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if api_key:
        return OpenAI(api_key=api_key.strip())

    # 2. Let the user enter their own key
    user_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key if the app's key is unavailable."
    )

    if user_key:
        return OpenAI(api_key=user_key.strip())

    # No API key available
    return None

openai_client = get_openai_client()

def get_api_token():
    """
    Retrieve a CourtListener API token.
    
    Priority:
    1. If COURTLISTENER_API_TOKEN exists in st.secrets, use that.
    2. Otherwise, allow the user to manually enter a token via the sidebar.
    """

    # 1. Try Streamlit secrets
    token = st.secrets.get("COURTLISTENER_API_TOKEN", None)
    if token:
        # Optional: uncomment if you want a confirmation that a secret was loaded
        # st.sidebar.caption(f"Using stored API token.")
        return token.strip()

    # 2. Ask user for a token if no stored secret
    token = st.sidebar.text_input(
        "CourtListener API Token",
        type="password",
        help="Enter your CourtListener API token if the app's token is unavailable."
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


def summarize_text(text: str, max_sentences: int = 7, max_chars: int = 1700):
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

def strip_html_tags(text: str) -> str:
    """Very small helper to remove basic HTML tags from snippets."""
    if not text:
        return ""
    # Remove tags like <b>...</b>, <em>, etc.
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def gpt_similarity_score(
    user_description: str,
    opinion_text: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
):
    if client is None:
        return None, "GPT is not configured."

    snippet = opinion_text[:2000]

    system_prompt = (
        "You are a legal research assistant. "
        "Given a user's case description and a court opinion, you rate how relevant "
        "the opinion is to the user's case on a 0 to 5 scale and explain why. "
        "0 = completely unrelated. 5 = very similar facts and legal issues. "
        "Respond ONLY in valid JSON with keys 'score' (number) and 'reason' (string)."
    )

    user_prompt = f"""
User's case:
\"\"\"{user_description}\"\"\"

Opinion excerpt:
\"\"\"{snippet}\"\"\"

JSON response example:
{{
  "score": 3,
  "reason": "short explanation here"
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

        # Extract text safely across SDK versions
        try:
            content = resp.output[0].content[0].text
        except Exception:
            content = getattr(resp, "output_text", str(resp))

        data = json.loads(content)
        score = float(data.get("score", 0))
        reason = data.get("reason", "")

        return score, reason

    except Exception as e:
        msg = str(e)

        if "insufficient_quota" in msg:
            return None, "GPT is unavailable (insufficient OpenAI quota). Using classical NLP only."

        if "429" in msg:
            return None, "GPT temporarily rate-limited. Using classical NLP only."

        return None, f"GPT error: {msg[:120]}"


# COURTLISTENER HELPERS (v4)
# -----------------------------
def search_case_metadata(
    token: str,
    query: str,
    jurisdiction: str = "all",
    max_results: int = 50,
) -> list[dict]:
    """
    Metadata-only search using CourtListener /search/ endpoint.
    Returns a list of dicts with id, case_name, citation, court, date, web_url, snippet.
    Over-fetches across pages up to max_results.
    """
    base_url = f"{BASE_URL}/search/"
    headers = make_headers(token)

    params = {
        "q": query,
        "type": "o",         # opinions
        "page_size": 20,     # page size per request; we loop pages
    }
    if jurisdiction != "all":
        params["jurisdiction"] = jurisdiction

    results: list[dict] = []
    next_url: str | None = base_url

    while next_url and len(results) < max_results:
        r = requests.get(next_url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()

        for item in data.get("results", []):
            opinion_id = item.get("id")
            abs_url = item.get("absolute_url") or ""
            if opinion_id is None and abs_url:
                m = re.search(r"/opinion/(\d+)/", abs_url)
                if m:
                    opinion_id = int(m.group(1))
            if opinion_id is None:
                # Skip if we really cannot determine the ID
                continue

            # --- Case name ---
            case_name = item.get("case_name") or item.get("caseName")

            if not case_name:
                cluster = item.get("cluster") or {}
                if isinstance(cluster, dict):
                    case_name = cluster.get("case_name") or cluster.get("caseName")

            if not case_name:
                docket = item.get("docket") or {}
                if isinstance(docket, dict):
                    case_name = docket.get("case_name") or docket.get("caseName")

            if not case_name and abs_url:
                m = re.search(r"/opinion/\d+/(.*?)/?$", abs_url)
                if m:
                    slug = m.group(1)
                    case_name = slug.replace("-", " ").title()
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
            date = (
                item.get("date_filed")
                or item.get("dateFiled")
                or item.get("date")
                or "Unknown date"
            )

            # --- Web URL ---
            if abs_url:
                if abs_url.startswith("http"):
                    web_url = abs_url
                else:
                    web_url = "https://www.courtlistener.com" + abs_url
            else:
                web_url = f"https://www.courtlistener.com/opinion/{opinion_id}/"

            # --- Snippet (metadata-level text, not full opinion) ---
            snippet = item.get("snippet", "")

            results.append(
                {
                    "id": opinion_id,
                    "case_name": case_name,
                    "citation": citation,
                    "court": court_name,
                    "date": date,
                    "web_url": web_url,
                    "snippet": snippet,
                }
            )

            if len(results) >= max_results:
                break

        # Pagination: CourtListener puts absolute next URL in "next"
        next_url = data.get("next")
        params = None  # subsequent requests follow next_url directly

    return results

def quick_metadata_filter(metadata_list, user_description):
    """
    Very cheap, robust filtering on metadata fields only.
    Returns a subset of metadata_list, or the full list if filtering is too strict.
    """

    if not metadata_list:
        return []

    # Extract quick keywords (light NLP)
    user_keywords = extract_keywords(user_description, top_n=10)
    user_keywords = [kw.lower() for kw in user_keywords]

    if not user_keywords:
        return metadata_list  # nothing to filter

    filtered = []

    for item in metadata_list:
        # Safe access
        case_name = (item.get("case_name") or "").lower()
        citation = (item.get("citation") or "").lower()
        court = (item.get("court") or "").lower()
        snippet = (item.get("snippet") or "").lower()

        # Very simple heuristics:
        # keep if ANY keyword appears in case_name, court, or snippet
        text_blob = " ".join([case_name, citation, court, snippet])

        if any(kw in text_blob for kw in user_keywords):
            filtered.append(item)

    # If too few after filtering, relax and return original metadata
    if len(filtered) < 5:
        return metadata_list

    return filtered

def get_case_snippets(token: str, metadata_cases: list[dict], max_chars: int = 800) -> list[dict]:
    """
    Given a list of metadata dicts (from search_case_metadata),
    attach a short 'snippet_text' to each case.

    Priority:
    1) Use the 'snippet' field from the search API when available.
    2) Otherwise, fetch the opinion text and take the first max_chars characters.

    Returns a new list of dicts with all original keys plus 'snippet_text'.
    """
    results = []

    for item in metadata_cases:
        opinion_id = item.get("id")
        snippet_raw = item.get("snippet") or ""

        # 1) Prefer the snippet returned by the search endpoint
        if snippet_raw.strip():
            snippet_text = strip_html_tags(snippet_raw)
        else:
            # 2) Fallback: fetch the beginning of the opinion text
            snippet_text = ""
            if opinion_id is not None:
                try:
                    # If you have a cached version, you can swap this to get_opinion_text_cached
                    full_text = get_opinion_text(token, opinion_id)
                    snippet_text = (full_text or "")[:max_chars]
                except Exception:
                    snippet_text = ""

        new_item = {**item}
        new_item["snippet_text"] = snippet_text
        results.append(new_item)

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

# filter cases only if they match all keywords
def build_search_query_from_description(desc: str, max_terms: int = 6) -> str:
    """
    Build a CourtListener search query from the user's description
    by extracting keywords and joining them with AND.
    If extraction fails, fall back to the raw description.
    """
    kws = extract_keywords(desc, top_n=max_terms)
    if not kws:
        return desc.strip()
    # e.g. "mold AND apartment AND landlord"
    return " AND ".join(kws)


def main():
    st.set_page_config(page_title="Legal Case Finder", layout="wide")

    st.title("🔎📚 Legal Case Finder")
    st.write(
        "Enter a legal scenario and the app retrieves similar cases using the CourtListener API."
    )

    st.sidebar.header("Settings")

    # Courtlistener and ChatGPT API
    token = get_api_token()
    if not token:
        st.sidebar.warning("Enter your CourtListener API token to begin.")
        st.stop()

    # ---------------- Jurisdiction UI ----------------
    preset_jurisdiction = st.sidebar.selectbox(
        "Choose or type a jurisdiction",
        options=["All", "New York (ny)", "California (ca)", "Federal (us)", "Custom..."],
        index=0,
        help=(
            "Pick a common jurisdiction or choose 'Custom...' to enter any "
            "CourtListener jurisdiction code (e.g., 'ny', 'ca', 'mass', 'us')."
        ),
    )

    jurisdiction_map = {
        "All": "all",
        "New York (ny)": "ny",
        "California (ca)": "ca",
        "Federal (us)": "us",
    }

    if preset_jurisdiction == "Custom...":
        custom_code = st.sidebar.text_input(
            "Custom jurisdiction code",
            value="",
            placeholder="e.g., ny, ca, mass, us",
            help="Type a valid CourtListener jurisdiction code. Leave blank for all.",
        )

        custom_code = custom_code.strip().lower()

        if not custom_code:
            st.sidebar.caption("No custom code entered → searching all jurisdictions.")
            jurisdiction = "all"
        else:
            if re.fullmatch(r"[a-z0-9_\-]+", custom_code):
                jurisdiction = custom_code
                st.sidebar.caption(f"Using custom jurisdiction code: '{jurisdiction}'")
            else:
                st.sidebar.warning(
                    "Jurisdiction codes should only contain letters, numbers, "
                    "hyphen, or underscore. Falling back to all jurisdictions."
                )
                jurisdiction = "all"
    else:
        jurisdiction = jurisdiction_map[preset_jurisdiction]

    # ---------------- Sidebar: other controls ----------------
    num_results = st.sidebar.slider(
        "Number of results",
        min_value=1,
        max_value=30, #this will look through 30 cases on COURTLISTENER; doesn't mean it will give back 30 cases
        value=5,
        step=1,
    )

    use_rerank = st.sidebar.checkbox(
        "Use NLP re-ranking (keyword overlap)",
        value=True,
        help="Rerank API results using keyword overlap between your description and each opinion.",
    )

    gpt_model = "gpt-4o-mini"  # cheapest model

    use_gpt_scoring = st.sidebar.checkbox(
        "Use GPT for Summary",
        value=False,
        help="Adds semantic relevance scoring when GPT quota is available.",
    )

    # ---------------- Main input ----------------
    st.subheader("Describe Your Case")
    user_desc = st.text_area(
        "Facts, issues, and context:",
        height=150,
        placeholder=(
            "Example: My client slipped on ice outside a supermarket, "
            "store knew about the hazard, premises liability, New York..."
        ),
    )

    if st.button("Search similar cases"):
        if not user_desc.strip():
            st.warning("Please enter a description first.")
            st.stop()

        # 1) Extract keywords from user description (for Jaccard + display)
        user_keywords = extract_keywords(user_desc, top_n=12)
        st.markdown("### Extracted keywords from your description")
        if user_keywords:
            st.write(", ".join(user_keywords))
        else:
            st.write("_No significant keywords detected (input may be too short)._")

        # 2) Build the CourtListener query string
        try:
            api_query = build_search_query_from_description(user_desc, max_terms=6)
            st.caption(f"Search query sent to CourtListener: `{api_query}`")
        except Exception as e:
            st.error(f"Error building search query: {e}")
            st.stop()

        # 3) METADATA-ONLY SEARCH (over-fetch)
        try:
            with st.spinner("Searching CourtListener (metadata only)..."):
                # You implement search_case_metadata to return a list of dicts with
                # id, case_name, citation, court, date, web_url, maybe a short snippet.
                metadata_results = search_case_metadata(
                    token=token,
                    query=api_query,
                    jurisdiction=jurisdiction,
                    max_results=num_results * 3,  # over-fetch
                )
        except Exception as e:
            st.error(f"Error while searching CourtListener: {e}")
            st.stop()

        if not metadata_results:
            st.info("No matching cases found.")
            st.stop()

        # 4) Quick metadata filter (e.g., by title, court, year)
        #    If filter is too strict and yields nothing, fall back to raw metadata.
        prelim_filtered = quick_metadata_filter(metadata_results, user_desc)
        if not prelim_filtered:
            prelim_filtered = metadata_results

        # 5) Get snippets (NOT full text) for Jaccard scoring
        #    Take a slightly reduced set for latency
        candidates_for_snippets = prelim_filtered[: num_results * 2]

        # You implement get_case_snippets to call CourtListener's "snippet" field,
        # or a very short text field, but NOT the full opinion.
        snippets = get_case_snippets(token, candidates_for_snippets)

        if not snippets:
            st.info("No snippets available for similarity scoring.")
            st.stop()

        # 6) Apply Jaccard filter on snippets
        jaccard_ranked = []
        for case in snippets:
            snippet_text = case.get("snippet_text", "") or ""
            if not snippet_text.strip():
                continue

            if use_rerank and user_keywords:
                case_tokens = tokenize(snippet_text)
                # require at least one overlapping keyword
                if not set(user_keywords) & set(case_tokens):
                    continue
                cheap_score = jaccard_score(user_keywords, case_tokens)
            else:
                cheap_score = None

            enriched = {**case}
            enriched["similarity"] = cheap_score
            enriched["gpt_score"] = None
            enriched["gpt_reason"] = ""
            enriched["text"] = ""        # will be filled with full text later
            enriched["summary"] = ""     # will be filled later
            jaccard_ranked.append(enriched)

        if not jaccard_ranked:
            st.info("No sufficiently similar cases found after filtering.")
            st.stop()

        if use_rerank:
            jaccard_ranked.sort(
                key=lambda x: (x["similarity"] if x["similarity"] is not None else 0.0),
                reverse=True,
            )

        # 7) Fetch FULL TEXT ONLY for TOP N after Jaccard
        top_n = jaccard_ranked[:num_results]

        for case in top_n:
            try:
                # You can wrap get_opinion_text with caching if you want:
                # full_text = get_opinion_text_cached(token, case["id"])
                full_text = get_opinion_text(token, case["id"])
            except Exception as e:
                full_text = f"Error retrieving opinion text: {e}"

            case["text"] = full_text
            case["summary"] = summarize_text(full_text, max_sentences=7)

        # 8) GPT reranking / scoring (only on top_n)
        if use_gpt_scoring:
            if openai_client is None:
                st.warning("GPT scoring enabled, but OPENAI_API_KEY is not configured in secrets.")
            else:
                # Only rerank the set we will actually show
                for case in top_n:
                    gpt_score, gpt_reason = gpt_similarity_score(
                        user_description=user_desc,
                        opinion_text=case["text"],
                        client=openai_client,
                        model=gpt_model,
                    )
                    time.sleep(0.7)  # avoid rate limit
                    case["gpt_score"] = gpt_score
                    case["gpt_reason"] = gpt_reason

                # Sort by GPT score (fallback -1 if missing)
                top_n.sort(
                    key=lambda x: (x["gpt_score"] if x["gpt_score"] is not None else -1),
                    reverse=True,
                )

        final_results = top_n  # what we display

        # 9) Display results
        st.markdown("## Results")

        for i, case in enumerate(final_results, start=1):
            similarity = case.get("similarity")
            sim_str = f"{similarity:.3f}" if similarity is not None else "N/A"

            gpt_score = case.get("gpt_score")
            gpt_reason = case.get("gpt_reason") or ""

            with st.expander(f"{i}. {case['case_name']}"):
                cols = st.columns([3, 2])
                with cols[0]:
                    st.markdown(f"**Citation:** {case.get('citation', 'N/A')}")
                    st.markdown(f"**Court:** {case.get('court', 'N/A')}")
                    st.markdown(f"**Date:** {case.get('date', 'N/A')}")
                    if gpt_score is not None:
                        st.markdown(f"**GPT Relevance score (0 – 5):** {gpt_score:.2f}")
                with cols[1]:
                    if case.get("web_url"):
                        st.markdown(
                            f"[Open full case on CourtListener]({case['web_url']})"
                        )

                if gpt_reason:
                    st.markdown("**GPT explanation of relevance:**")
                    st.write(gpt_reason)

                st.markdown("**First few sentences of opinion:**")
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
