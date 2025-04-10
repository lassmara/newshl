import streamlit as st
import pandas as pd
import re
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# -------------------- CONFIG --------------------
st.set_page_config(page_title="🔍 SHL Job Assessment Recommender", layout="wide")
st.title("🤖 SHL Assessment Recommender")
st.markdown("Enter a job description to discover matching SHL assessments.")

# -------------------- GEMINI SETUP --------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_raw_data():
    df = pd.read_csv("shl_detailed_enriched.csv")
    df["duration_minutes"] = df["duration"].str.extract(r'(\d+)', expand=False).astype(float)
    df["description"] = df["description"].fillna("").str.replace(r'\s+', ' ', regex=True)
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")






# -------------------- GEMINI FUNCTIONS --------------------
def enhance_query_with_gemini(raw_query):
    try:
        prompt = (
            "SHL Test Types:\n"
            "A - Ability & Aptitude\n"
            "B - Biodata & Situational Judgement\n"
            "C - Competencies\n"
            "D - Development & 360\n"
            "E - Assessment Exercises\n"
            "K - Knowledge & Skills\n"
            "P - Personality & Behavior\n"
            "S - Simulations\n\n"
            "Valid options for classification:\n"
            "- Job Family: Business, Clerical, Contact Center, Customer Service, Information Technology, Safety, Sales\n"
            "- Job Level: Director, Entry-Level, Executive, Front Line Manager, General Population, Graduate, Manager, Mid-Professional, Professional Individual Contributor, Supervisor\n"
            "- Industry: Banking/Finance, Healthcare, Hospitality, Insurance, Manufacturing, Oil & Gas, Retail, Telecommunications\n"
            "- Language: English, Romanian, (Brazil)\n\n"
            f"Job Description:\n{raw_query}\n\n"
            "1. Rewrite the job description into a structured summary.\n"
            "2. Return SHL test types in set format (e.g., {A, K, P}).\n"
            '3. Return metadata as Python dict format like this:\n'
            '{"Job Family": "Information Technology", "Job Level": "Graduate", "Industry": "Banking/Finance", "Language": "English"}'
        )
        model = genai.GenerativeModel(model_name="gemini-2.5-pro-exp-03-25")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return raw_query

def extract_test_types_and_metadata(text):
    test_types = []
    metadata = {}

    try:
        # Extract set: {A, K, P}
        match = re.search(r'\{([^}]+)\}', text)
        if match:
            raw_codes = [c.strip() for c in match.group(1).split(",")]
            valid_codes = {'A', 'B', 'C', 'D', 'E', 'K', 'P', 'S'}
            test_types = [c for c in raw_codes if c in valid_codes]
    except Exception as e:
        st.warning(f"Error parsing test types: {e}")

    try:
        # Extract dict
        dict_match = re.search(r'\{[^{}]*"Job Family"[^{}]+\}', text)
        if dict_match:
            metadata = ast.literal_eval(dict_match.group())
    except Exception as e:
        st.warning(f"Error parsing metadata: {e}")

    return test_types, metadata

# -------------------- UI --------------------

query = st.text_area("Job description or query:", height=150, placeholder="e.g., Hiring for a frontend engineer with JavaScript skills...")
top_k = st.slider("🔢 Number of recommendations to show:", min_value=1, max_value=20, value=10)
max_duration = st.number_input("⏱️ Max duration (in minutes):", min_value=5, max_value=120, value=60)


with st.spinner("Loading data and generating embeddings..."):
    df = load_raw_data()
    model = load_model()
    df["embedding"] = df["description"].apply(lambda x: model.encode(x, show_progress_bar=False))
    
st.success("✅ Data loaded and embeddings generated.")


if st.button("Search") and query.strip():
    with st.spinner("Analyzing ..."):
        gemini_response = enhance_query_with_gemini(query)
        test_types, metadata = extract_test_types_and_metadata(gemini_response)

    # st.markdown("### 🧠 Gemini Response")
    # st.code(gemini_response)

    st.markdown("#### 📌 Detected SHL Test Types")
    st.write(", ".join(test_types) if test_types else "None")

    st.markdown("#### 🗂️ Inferred Metadata")
    st.json(metadata)

    # -------------------- FILTER BY DURATION --------------------
    filtered_df = df[df["duration_minutes"] <= max_duration].copy()
    query_vec = model.encode([gemini_response])

    def calculate_score(row):
        score = cosine_similarity([row["embedding"]], query_vec)[0][0]

        # Boost for test type match
        if any(code in str(row["test_types"]) for code in test_types):
            score += 0.5

        # Metadata matches (only if you later add these columns in your CSV)
        if "role" in row and metadata.get("Job Level") and metadata["Job Level"].lower() in str(row["role"]).lower():
            score += 0.3
        if "role" in row and metadata.get("Job Family") and metadata["Job Family"].lower() in str(row["role"]).lower():
            score += 0.3

        return score

    filtered_df["total_score"] = filtered_df.apply(calculate_score, axis=1)

    results = filtered_df.sort_values(by="total_score", ascending=False).head(top_k).copy()
    results["role"] = results.apply(lambda row: f"[{row['role']}]({row['link']})", axis=1)
    results["duration"] = results["duration_minutes"].apply(lambda x: f"{x:.1f} min" if pd.notna(x) else "")

    display_cols = ["role", "duration", "test_types", "remote_testing", "adaptive_irt", "description"]
    col_labels = {
        "role": "🧑‍💼 Role",
        "duration": "⏱ Duration",
        "test_types": "🧪 Types",
        "remote_testing": "🌐 Remote?",
        "adaptive_irt": "📈 Adaptive?",
        "description": "📝 Description"
    }

    st.markdown("### ✅ Top Recommended Assessments")
    st.dataframe(results[display_cols].rename(columns=col_labels), use_container_width=True)
    result_json = results[display_cols].rename(columns=col_labels).to_dict(orient='records')
    st.json(result_json)