import re
import streamlit as st
import requests
import json
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words("english"))

base_url = "http://127.0.0.1:8000/"
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .main {
        display: flex;
        justify-content: center;
        padding-top: 30px;
    }
    .content-container {
        max-width: 1200px; /* Adjust width for centering */
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="content-container">', unsafe_allow_html=True)
st.title('_:blue[Local File Search]_ :📂')

if "search_result" not in st.session_state:
    st.session_state.search_result = []
if "ai_result" not in st.session_state:
    st.session_state.ai_result = ""
if "search_input" not in st.session_state:
    st.session_state.search_input = ""
if "ai_input" not in st.session_state:
    st.session_state.ai_input = ""

def format_keywords_as_list(content, keywords, num_words=10):
    filtered_keywords = [kw for kw in keywords if kw.lower() not in STOPWORDS]
    escaped_keywords = "|".join(map(re.escape, filtered_keywords))
    
    if not escaped_keywords:
        return ["No relevant content found."]

    matches = list(re.finditer(escaped_keywords, content, re.IGNORECASE))
    if not matches:
        return ["No matches found."]

    snippets = []
    for match in matches:
        start_index = match.start()
        words_before = content[:start_index].split()[-10:]
        words_after = content[start_index:].split()[:num_words + 1]
        snippet = " ".join(words_before + words_after)

        highlighted_snippet = re.sub(
            escaped_keywords,
            lambda m: f"<span style='background-color: yellow; font-weight: bold;'>{m.group(0)}</span>",
            snippet,
            flags=re.IGNORECASE,
        )
        snippets.append(f"... {highlighted_snippet} ...")
    
    return snippets

left_col, right_col = st.columns([1, 1])
with left_col:
    st.subheader("Search Files")
    search_input = st.text_input("Enter keywords to search your local files:", st.session_state.search_input, key="search_input_key")
    if st.button("Search files"):
        st.session_state.search_input = search_input
        url = "http://127.0.0.1:8000/search"

        payload = json.dumps({"query": search_input})
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            response_data = response.json()

            if isinstance(response_data, list):
                st.session_state.search_result = response_data
            else:
                st.session_state.search_result = [{"content": "Unexpected data format received.", "path": ""}]

        except requests.exceptions.RequestException as e:
            st.session_state.search_result = [{"content": f"HTTP Request failed: {e}", "path": ""}]
        except json.JSONDecodeError:
            st.session_state.search_result = [{"content": "Failed to decode JSON response.", "path": ""}]

    if st.session_state.search_result:
        st.write("### Results:")
        for item in st.session_state.search_result:
            keywords = st.session_state.search_input.split()
            snippets = format_keywords_as_list(item.get('content', ""), keywords)
            
            valid_snippets = [snippet for snippet in snippets if snippet != "No matches found."]
            if valid_snippets:
                item_path = item.get('path', 'Unknown File').replace('TestFolder', 'files')
                st.markdown(f"<span style='font-size:20px; font-weight:bold;'>Document: <a href='{base_url}{item_path}' target='_blank' style='text-decoration: none; color: blue;'>{item_path}</a></span>",
                            unsafe_allow_html=True)
                for snippet in valid_snippets:
                    st.markdown(f"- {snippet}", unsafe_allow_html=True)


with right_col:
    st.subheader("📂 File Links Preview")
    st.write("Click on a file link to preview:")
    
    file_links = [
        {"name": "2201.01647v4.pdf", "url": f"{base_url}files/2201.01647v4.pdf"},
        {"name": "SeraphimdroidEmail.txt", "url": f"{base_url}files/subfolder/SeraphimdroidEmail.txt"},
    ]

    for file in file_links:
        st.markdown(
            f"<a href='{file['url']}' target='_blank' style='font-size:16px; color: blue; text-decoration: none;'>"
            f"📄 {file['name']}</a>",
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)