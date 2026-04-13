import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from metrics_store import DB_PATH

st.set_page_config(page_title="RAG monitor", layout="wide")
st.title("RAG pipeline monitor")


@st.cache_data(ttl=10)
def load():
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql("SELECT * FROM request_metrics ORDER BY ts DESC", c)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    df["cost_usd"] = (
        df["prompt_tokens"] * 0.0000025 +
        df["completion_tokens"] * 0.000010
    )
    return df


df = load()

if df.empty:
    st.info("No requests yet — run some queries through rag.py first.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requests", len(df))
c2.metric("P50 latency", f"{df['total_ms'].quantile(0.50):.0f} ms")
c3.metric("P95 latency", f"{df['total_ms'].quantile(0.95):.0f} ms")
c4.metric("Grounded", f"{df['grounded'].mean()*100:.1f}%")
c5.metric("Total cost", f"${df['cost_usd'].sum():.4f}")

st.divider()

st.subheader("End-to-end latency over time")
fig = px.line(df.sort_values("datetime"), x="datetime", y="total_ms",
              labels={"total_ms": "Total latency (ms)", "datetime": ""})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Avg latency by stage")
stage_df = pd.DataFrame({
    "stage": ["retrieval", "generation"],
    "avg_ms": [df["ret_ms"].mean(), df["gen_ms"].mean()]
})
fig2 = px.bar(stage_df, x="stage", y="avg_ms",
              labels={"avg_ms": "Avg (ms)", "stage": ""})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Grounded rate (rolling 10 requests)")
df_sorted = df.sort_values("datetime").copy()
df_sorted["grounded_roll"] = df_sorted["grounded"].rolling(10, min_periods=1).mean() * 100
fig3 = px.line(df_sorted, x="datetime", y="grounded_roll",
               labels={"grounded_roll": "Grounded % (rolling 10)", "datetime": ""})
fig3.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="threshold (75%)")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Cost per request (USD)")
fig4 = px.histogram(df, x="cost_usd", nbins=20,
                    labels={"cost_usd": "Cost (USD)"})
st.plotly_chart(fig4, use_container_width=True)

st.subheader("Recent requests")
st.dataframe(
    df[["datetime", "question", "total_ms", "ret_ms", "gen_ms", "grounded", "cost_usd"]]
    .head(50).reset_index(drop=True),
    use_container_width=True
)
