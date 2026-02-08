import os
from constant import OPENAI_API_KEY
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Ai Lead Qualification Agent")

mode = st.radio(
     "Choose input mode:",
     ["Upload CSV", "Single company lookup"]
)


def scrape_website(url):
        try:
            r = requests.get(url, timeout=5)
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text[:2000]
        except:
            return "No website content found"

        
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are an AI sales analyst scoring leads against an Ideal Customer Profile.

ICP:
- B2B SaaS company
- 20 to 500 employees
- Growing sales organization
- Buyer is VP Sales / RevOps / Founder
- Modern tech company

Lead:
Name: {name}
Company: {company}
Title: {title}
Website summary: {summary}

Score the lead from 1 to 10.

Return ONLY valid JSON:
{{
  "score": number,
  "reason": "short explanation"
}}
""")

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded leads:")
        st.dataframe(df.head(), hide_index=True)

        run_csv = st.button("Run CSV Agent")

        if run_csv:
            results = []

            with st.spinner("Scoring leads..."):
                for _, row in df.iterrows():
                    summary = scrape_website(row["website"])

                    chain = prompt | llm
                    response = chain.invoke({
                        "name": row["name"],
                        "company": row["company"],
                        "title": row["title"],
                        "summary": summary
                    })

                    try:
                        parsed = json.loads(response.content)
                        score = parsed["score"]
                        reason = parsed["reason"]
                    except:
                        score = 0
                        reason = "Failed to parse AI output"

                   
                    if score >= 7:
                        category = "Very Suitable"
                    elif score >= 4:
                        category = "Moderate Match"
                    else:
                        category = "Low Match"

                    results.append({
                        "name": row["name"],
                        "company": row["company"],
                        "score": score,
                        "category": category,
                        "reason": reason
                    })

            results_df = pd.DataFrame(results)

            high = results_df[results_df["score"] >= 7]
            mid = results_df[(results_df["score"] >= 4) & (results_df["score"] <= 6)]
            low = results_df[results_df["score"] <= 3]

            high = high.sort_values(by="score", ascending=False)
            mid = mid.sort_values(by="score", ascending=False)
            low = low.sort_values(by="score", ascending=False)

            st.subheader("Very Suitable Leads (7–10)")
            st.dataframe(high, hide_index=True)

            st.subheader("Moderate Match (4–6)")
            st.dataframe(mid, hide_index=True)

            st.subheader("Low Match (1–3)")
            st.dataframe(low, hide_index=True)


if mode == "Single company lookup":
    name = st.text_input("Name")
    company = st.text_input("Company")
    title = st.text_input("Title")
    website = st.text_input("Website URL")

    run_lookup = st.button("Run Lookup Agent")

    if run_lookup and website:
        with st.spinner("Analyzing company..."):
            summary = scrape_website(website)

            chain = prompt | llm
            response = chain.invoke({
                "name": name,
                "company": company,
                "title": title,
                "summary": summary
            })

        try:
            parsed = json.loads(response.content)
            score = parsed["score"]
            reason = parsed["reason"]
        except:
            score = 0
            reason = "Failed to parse AI output"

        if score >= 7:
            category = "Very Suitable"
        elif score >= 4:
            category = "Moderate Match"
        else:
            category = "Low Match"

        st.markdown(f"## Score: {score}/10")
        st.markdown(f"### Category: {category}")
        st.markdown("**Reasoning:**")
        st.write(reason)