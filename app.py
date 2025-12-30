import os
import json

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------
# 2. Helper functions
# -----------------------
EXTRACTION_PROMPT = """
You are a forensic accounting assistant analyzing Indian IPO prospectuses (DRHP/RHP).

Given the full text of a DRHP, extract the following fields if available.
Return ONLY valid JSON, no commentary.

Fields:
{
  "company_name": string or null,
  "ipo_year": number or null,
  "sector": string or null,

  "financials": {
    "years": [Y3, Y2, Y1],
    "revenue": [r3, r2, r1],
    "ebitda_margin_pct": [m3, m2, m1],
    "trade_receivables": [tr3, tr2, tr1],
    "rpt_revenue_pct": [rp3, rp2, rp1]
  },

  "use_of_proceeds": [
    {
      "category": "Debt repayment" | "Capex" | "Working capital" | "General corporate purposes" | "Acquisitions" | "Other",
      "amount_crore": number
    }
  ],

  "key_risk_factors": [
    short bullet points (max 7) summarising the most material risk factors mentioned
  ]
}

Rules:
- If any data is missing or unclear, use null or empty arrays.
- Convert all currency to crore INR where possible.
- Do NOT include any keys other than the ones specified.
- Output MUST be a single JSON object.
"""


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        if i >= 40:
            break
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def get_structured_data(text: str) -> dict:
    prompt = EXTRACTION_PROMPT + "\n\nDRHP TEXT:\n\n" + text
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Clean possible ```
# if raw.startswith("```"):
#     raw = raw.replace("`", "")
#     if raw.lower().startswith("json"):
#         raw = raw[4:].strip()


    # Ensure we only take the JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    data = json.loads(raw[start : end + 1])
    return data


def compute_flags(data: dict) -> dict:
    fin = data.get("financials") or {}
    years = fin.get("years") or []
    rev = fin.get("revenue") or []
    rec = fin.get("trade_receivables") or []
    rpt = fin.get("rpt_revenue_pct") or []
    uop = data.get("use_of_proceeds") or []

    flags = {}

    def cagr(series):
        # Handle empty / too short
        if not series or len(series) < 2:
            return None

        # Flatten or extract numeric values
        cleaned = []
        for x in series:
            # If each element is like [year, value] or ("2021", 1000)
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                value = x[1]
            else:
                value = x

            # Skip None / non‑numeric entries
            if value is None:
                continue
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError):
                continue

        # Need at least first and last numeric value
        if len(cleaned) < 2:
            return None

        first, last = cleaned[0], cleaned[-1]
        if first <= 0 or last <= 0:
            return None

        n = len(cleaned) - 1
        return (last / first) ** (1.0 / n) - 1.0

    rev_cagr = cagr(rev)
    rec_cagr = cagr(rec)
    flags["revenue_cagr"] = rev_cagr
    flags["receivables_cagr"] = rec_cagr
    flags["receivables_grow_faster"] = (
        rev_cagr is not None and rec_cagr is not None and rec_cagr > rev_cagr + 0.05
    )

    if rev and rec and rev[-1] and rec[-1]:
        flags["rec_to_rev_latest"] = rec[-1] / rev[-1]
    else:
        flags["rec_to_rev_latest"] = None

    if rpt and rpt[-1] is not None:
        flags["rpt_share_latest"] = rpt[-1]
        flags["rpt_share_high"] = rpt[-1] > 20
    else:
        flags["rpt_share_latest"] = None
        flags["rpt_share_high"] = False

    total_proc = 0.0
    gcp = 0.0
    for item in uop:
        amt = item.get("amount_crore")
        cat = (item.get("category") or "").lower()
        if amt is None:
            continue
        total_proc += amt
        if "general" in cat:
            gcp += amt
    gcp_pct = (gcp / total_proc * 100) if total_proc > 0 else None
    flags["gcp_pct"] = gcp_pct
    flags["gcp_high"] = gcp_pct is not None and gcp_pct > 30

    return flags


def get_commentary(data: dict, flags: dict) -> str:
    prompt = f"""
You are a forensic accounting expert assessing an Indian IPO.

STRUCTURED DATA:
{json.dumps(data, indent=2)}

FLAGS:
{json.dumps(flags, indent=2)}

TASK:
Write a concise 10-15 sentence assessment of this IPO from a financial-statement-forensic perspective, with suggestions on whether it's a good investment or not.
Comment specifically on, any possibility of accounting manipulation or misrepresentation, financial health of company, receivables vs revenue growth, related-party revenue dependence, and use of proceeds.
If some data is missing, mention that briefly rather than guessing.
Return only the paragraph.
"""
    response = model.generate_content(prompt)
    return response.text.strip()


# -----------------------
# 3. Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="IPO Forensic DRHP Analyzer", layout="wide")
    st.title("IPO Forensic Checklist (DRHP Analyzer)")

    uploaded_file = st.file_uploader("Upload DRHP/RHP PDF", type=["pdf"])

    if uploaded_file and st.button("Analyze"):
        with st.spinner("Reading PDF..."):
            text = extract_pdf_text(uploaded_file)

        with st.spinner("Extracting structured data using AI..."):
            data = get_structured_data(text)

        with st.spinner("Computing red flags..."):
            flags = compute_flags(data)

        with st.spinner("Generating commentary..."):
            comment = get_commentary(data, flags)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Info")
            st.write(f"**Company:** {data.get('company_name')}")
            st.write(f"**IPO Year:** {data.get('ipo_year')}")
            st.write(f"**Sector:** {data.get('sector')}")
            st.subheader("Financials (raw JSON)")
            st.json(data.get("financials") or {})
            st.subheader("Use of Proceeds")
            st.json(data.get("use_of_proceeds") or [])

        with col2:
            st.subheader("Red-Flag Metrics")
            st.write(flags)
            st.subheader("AI Forensic Commentary")
            st.write(comment)

        st.subheader("Key Risk Factors")
        for rf in data.get("key_risk_factors") or []:
            st.write(f"- {rf}")


if __name__ == "__main__":
    main()



