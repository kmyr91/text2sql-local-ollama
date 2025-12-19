import time
import requests
import pandas as pd
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Text2SQL (Local)", layout="wide")
st.title("Text2SQL Assistant (Local / Free)")
st.write("Type a question → the app generates SQL → runs it on Postgres (Chinook) → returns results.")

EXAMPLES = [
    "Top 5 artists by total revenue",
    "Top 5 customers by total invoice amount",
    "Revenue by billing country (top 10)",
    "Top 10 tracks by number of purchases",
    "Which genres have the most tracks?",
]

# Sidebar
st.sidebar.header("Examples")
picked = st.sidebar.selectbox("Pick an example", EXAMPLES)
use_example = st.sidebar.button("Use example")

st.sidebar.divider()
show_schema = st.sidebar.checkbox("Show schema (debug)", value=False)

if "history" not in st.session_state:
    st.session_state.history = []

# Main input
default_q = picked if use_example else "Top 5 artists by total revenue"
question = st.text_input("Your question", value=default_q)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run_btn = st.button("Run", type="primary")
with col2:
    clear_btn = st.button("Clear history")
with col3:
    st.caption("Keep the FastAPI server running while using this UI.")

if clear_btn:
    st.session_state.history = []
    st.success("History cleared.")

if show_schema:
    r = requests.get(f"{API_BASE}/schema", timeout=30)
    r.raise_for_status()
    st.subheader("Schema + Foreign Keys")
    st.code(r.json()["schema_text"], language="text")

def render_result(payload: dict, elapsed: float):
    st.caption(f"Completed in {elapsed:.2f}s")

    st.subheader("Generated SQL")
    st.code(payload["sql"], language="sql")

    cols = payload["result"]["columns"]
    rows = payload["result"]["rows"]
    df = pd.DataFrame(rows, columns=cols)

    st.subheader("Result")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="result.csv",
        mime="text/csv",
    )

    # Quick chart if it looks like (label, number)
    if df.shape[1] >= 2:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            y = numeric_cols[0]
            x = [c for c in df.columns if c != y][0]
            chart_df = df[[x, y]].copy().set_index(x)
            st.subheader("Quick chart")
            st.bar_chart(chart_df)

    if "note" in payload:
        st.info(payload["note"])

if run_btn:
    start = time.time()
    try:
        r = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=300)
        elapsed = time.time() - start

        if r.status_code != 200:
            st.error("Query failed")
            st.json(r.json())
        else:
            payload = r.json()
            render_result(payload, elapsed)

            # Save to history
            cols = payload["result"]["columns"]
            rows = payload["result"]["rows"]
            df = pd.DataFrame(rows, columns=cols)
            st.session_state.history.insert(0, {"q": question, "sql": payload["sql"], "df": df})

    except Exception as e:
        st.error(str(e))

# History section
if st.session_state.history:
    st.divider()
    st.subheader("History (this session)")
    for i, item in enumerate(st.session_state.history[:10], start=1):
        with st.expander(f"{i}. {item['q']}"):
            st.code(item["sql"], language="sql")
            st.dataframe(item["df"], use_container_width=True)
