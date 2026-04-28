import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Orchestrator Approval Dashboard",
    page_icon="📊",
    layout="wide",
)

APPROVED_KEYWORDS = ["approved", "success", "successful", "paid", "completed", "captured", "settled"]
DECLINED_KEYWORDS = ["declined", "failed", "rejected", "error", "cancelled", "canceled", "expired", "aborted"]

COLUMN_ALIASES = {
    "psp": ["pspName", "psp", "provider", "paymentProvider", "processor"],
    "country": ["country", "cardCountry", "customerCountry", "billingCountry"],
    "merchant_order_id": ["merchantOrderId", "merchant_order_id", "merchant order id", "orderId", "merchantOrderID"],
    "status": ["status", "transactionStatus", "state"],
    "decline_reason": ["declineReason", "decline_reason", "reason", "errorReason", "gatewayDeclineReason"],
    "decline_code": ["declineCode", "decline_code", "errorCode", "responseCode"],
    "mid": ["midAlias", "mid", "MID", "merchantId", "merchant_id"],
    "amount": ["amount", "transactionAmount"],
    "currency": ["currency", "transactionCurrency"],
    "date": ["processing_date", "processingDate", "completionDate", "createdAt", "created_at", "date"],
}


def find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lower_map:
            return lower_map[key]
    # fallback partial match
    for alias in aliases:
        key = alias.strip().lower().replace("_", "")
        for col in df.columns:
            normalized = str(col).strip().lower().replace("_", "").replace(" ", "")
            if key == normalized:
                return col
    return None


def normalize_status(value: object) -> str:
    text = str(value).strip().lower()
    if any(k in text for k in APPROVED_KEYWORDS):
        return "Approved"
    if any(k in text for k in DECLINED_KEYWORDS):
        return "Declined"
    if text in ["nan", "none", ""]:
        return "Unknown"
    return "Other"


def classify_payment_type(psp: object) -> str:
    text = str(psp).strip().lower()
    if "confirmo" in text:
        return "Crypto"
    if "paypal" in text or "pay pal" in text:
        return "P2P"
    return "International Card"


def safe_ratio(numerator, denominator):
    if denominator in [0, None] or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator) * 100


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False, encoding_errors="replace")


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mapping = {key: find_col(df, aliases) for key, aliases in COLUMN_ALIASES.items()}
    missing = [k for k in ["psp", "merchant_order_id", "status"] if mapping.get(k) is None]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}. Please check the uploaded file headers.")
        st.stop()

    out = pd.DataFrame()
    out["psp"] = df[mapping["psp"]].astype(str).str.strip()
    out["merchant_order_id"] = df[mapping["merchant_order_id"]].astype(str).str.strip()
    out["status_raw"] = df[mapping["status"]].astype(str).str.strip()
    out["status_group"] = out["status_raw"].apply(normalize_status)
    out["payment_type"] = out["psp"].apply(classify_payment_type)

    out["country"] = df[mapping["country"]].astype(str).str.strip() if mapping.get("country") else "Unknown"
    out["mid"] = df[mapping["mid"]].astype(str).str.strip() if mapping.get("mid") else "Unknown"
    out["decline_reason"] = df[mapping["decline_reason"]].astype(str).str.strip() if mapping.get("decline_reason") else "Unknown"
    out["decline_code"] = df[mapping["decline_code"]].astype(str).str.strip() if mapping.get("decline_code") else "Unknown"
    out["amount"] = pd.to_numeric(df[mapping["amount"]], errors="coerce") if mapping.get("amount") else np.nan
    out["currency"] = df[mapping["currency"]].astype(str).str.strip() if mapping.get("currency") else "Unknown"

    if mapping.get("date"):
        out["txn_datetime"] = pd.to_datetime(df[mapping["date"]], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        out["txn_datetime"] = pd.NaT
    out["txn_date"] = out["txn_datetime"].dt.date

    out = out[out["merchant_order_id"].notna() & (out["merchant_order_id"] != "") & (out["merchant_order_id"].str.lower() != "nan")]
    return out, mapping


def unique_order_summary(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    order_level = (
        data.groupby(group_cols + ["merchant_order_id"], dropna=False)
        .agg(
            attempts=("merchant_order_id", "size"),
            approved=("status_group", lambda x: int((x == "Approved").any())),
            declined_only=("status_group", lambda x: int(not (x == "Approved").any())),
        )
        .reset_index()
    )
    summary = (
        order_level.groupby(group_cols, dropna=False)
        .agg(
            unique_orders=("merchant_order_id", "nunique"),
            approved_orders=("approved", "sum"),
            declined_unique_orders=("declined_only", "sum"),
            total_attempts=("attempts", "sum"),
            retried_orders=("attempts", lambda x: int((x > 1).sum())),
            avg_attempts_per_order=("attempts", "mean"),
        )
        .reset_index()
    )
    summary["approval_ratio_%"] = summary.apply(lambda r: safe_ratio(r["approved_orders"], r["unique_orders"]), axis=1)
    summary["retry_order_ratio_%"] = summary.apply(lambda r: safe_ratio(r["retried_orders"], r["unique_orders"]), axis=1)
    summary["retry_attempt_ratio_%"] = summary.apply(lambda r: safe_ratio(r["total_attempts"] - r["unique_orders"], r["total_attempts"]), axis=1)
    return summary.sort_values(["approval_ratio_%", "unique_orders"], ascending=[False, False])


def build_routing(data: pd.DataFrame, min_orders: int) -> pd.DataFrame:
    base = unique_order_summary(data, ["country", "psp"])
    if base.empty:
        return base
    base = base[base["unique_orders"] >= min_orders].copy()
    if base.empty:
        return base
    best = base.sort_values(["country", "approval_ratio_%", "unique_orders"], ascending=[True, False, False])
    best = best.groupby("country", as_index=False).first()
    best = best.rename(columns={"psp": "recommended_psp", "approval_ratio_%": "recommended_approval_%", "unique_orders": "recommended_unique_orders"})
    country_total = unique_order_summary(data, ["country"])[["country", "unique_orders", "approved_orders", "approval_ratio_%"]]
    country_total = country_total.rename(columns={"approval_ratio_%": "current_country_approval_%"})
    rec = best.merge(country_total, on="country", how="left")
    rec["insight"] = np.where(
        rec["recommended_approval_%"] > rec["current_country_approval_%"],
        "Route more volume to the recommended PSP, subject to risk and cost checks.",
        "Current country mix is already close to the best observed PSP performance.",
    )
    return rec.sort_values(["recommended_approval_%", "recommended_unique_orders"], ascending=[False, False])


st.title("📊 Orchestrator Approval, Retry & Routing Dashboard")
st.caption("Approval ratio is calculated on unique Merchant Order ID. Confirmo = Crypto, PayPal = P2P, all other PSPs = International Card.")

with st.sidebar:
    st.header("Upload & Filters")
    uploaded = st.file_uploader("Upload orchestrator CSV", type=["csv"])
    min_orders = st.number_input("Minimum unique orders for routing recommendation", min_value=1, value=10, step=1)

if uploaded is None:
    st.info("Upload your orchestrator CSV report to start the dashboard.")
    st.stop()

raw = load_csv(uploaded.read())
data, mapping = prepare_data(raw)

with st.sidebar:
    if data["txn_date"].notna().any():
        min_date = data["txn_date"].dropna().min()
        max_date = data["txn_date"].dropna().max()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None

    payment_types = st.multiselect("Payment type", sorted(data["payment_type"].dropna().unique()), default=sorted(data["payment_type"].dropna().unique()))
    countries = st.multiselect("Country", sorted(data["country"].dropna().unique()))
    psps = st.multiselect("PSP", sorted(data["psp"].dropna().unique()))
    mids = st.multiselect("MID", sorted(data["mid"].dropna().unique()))

filtered = data.copy()
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["txn_date"].isna()) | ((filtered["txn_date"] >= start) & (filtered["txn_date"] <= end))]
elif date_range and isinstance(date_range, date):
    # User selected only a single date — treat as single-day filter
    filtered = filtered[(filtered["txn_date"].isna()) | (filtered["txn_date"] == date_range)]
if payment_types:
    filtered = filtered[filtered["payment_type"].isin(payment_types)]
if countries:
    filtered = filtered[filtered["country"].isin(countries)]
if psps:
    filtered = filtered[filtered["psp"].isin(psps)]
if mids:
    filtered = filtered[filtered["mid"].isin(mids)]

order_level = (
    filtered.groupby("merchant_order_id")
    .agg(attempts=("merchant_order_id", "size"), approved=("status_group", lambda x: int((x == "Approved").any())))
    .reset_index()
)
unique_orders = int(order_level["merchant_order_id"].nunique()) if not order_level.empty else 0
approved_orders = int(order_level["approved"].sum()) if not order_level.empty else 0
total_attempts = int(len(filtered))
retried_orders = int((order_level["attempts"] > 1).sum()) if not order_level.empty else 0
approval_ratio = safe_ratio(approved_orders, unique_orders)
retry_order_ratio = safe_ratio(retried_orders, unique_orders)
retry_attempt_ratio = safe_ratio(total_attempts - unique_orders, total_attempts)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Unique Orders", f"{unique_orders:,}")
k2.metric("Approved Unique Orders", f"{approved_orders:,}")
k3.metric("Approval Ratio", f"{approval_ratio:.2f}%")
k4.metric("Retry Order Ratio", f"{retry_order_ratio:.2f}%")
k5.metric("Retry Attempt Ratio", f"{retry_attempt_ratio:.2f}%")

st.divider()

st.subheader("Executive Updates")
psp_perf = unique_order_summary(filtered, ["psp"])
country_perf = unique_order_summary(filtered, ["country"])
routing = build_routing(filtered, int(min_orders))

def top_line_insights():
    lines = []
    if not psp_perf.empty:
        best = psp_perf.iloc[0]
        worst_candidates = psp_perf[psp_perf["unique_orders"] >= max(3, int(min_orders))]
        worst = worst_candidates.sort_values("approval_ratio_%").iloc[0] if not worst_candidates.empty else psp_perf.sort_values("approval_ratio_%").iloc[0]
        lines.append(f"Best PSP by unique-order approval: **{best['psp']}** at **{best['approval_ratio_%']:.2f}%** from **{int(best['unique_orders']):,}** unique orders.")
        lines.append(f"Lowest PSP approval with meaningful volume: **{worst['psp']}** at **{worst['approval_ratio_%']:.2f}%** from **{int(worst['unique_orders']):,}** unique orders.")
    if not country_perf.empty:
        country_low = country_perf[country_perf["unique_orders"] >= max(3, int(min_orders))].sort_values("approval_ratio_%").head(1)
        if not country_low.empty:
            r = country_low.iloc[0]
            lines.append(f"Country needing routing review: **{r['country']}** with **{r['approval_ratio_%']:.2f}%** approval from **{int(r['unique_orders']):,}** unique orders.")
    if retry_order_ratio > 25:
        lines.append(f"Retry pressure is high at **{retry_order_ratio:.2f}%** of unique orders. Review cascading rules, issuer response mapping, and repeated attempts from same Merchant Order ID.")
    if routing is not None and not routing.empty:
        r = routing.iloc[0]
        lines.append(f"Strongest routing opportunity: **{r['country']} → {r['recommended_psp']}** with observed approval of **{r['recommended_approval_%']:.2f}%**.")
    return lines

for line in top_line_insights():
    st.markdown(f"- {line}")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "PSP Analysis",
    "Country Analysis",
    "Decline Reasons",
    "MID Analysis",
    "Routing Insights",
    "Raw Data",
])

with tab1:
    c1, c2 = st.columns(2)
    daily = unique_order_summary(filtered.dropna(subset=["txn_date"]), ["txn_date"])
    if not daily.empty:
        fig = px.line(daily, x="txn_date", y="approval_ratio_%", markers=True, title="Date-wise Approval Ratio")
        c1.plotly_chart(fig, use_container_width=True)
    type_summary = unique_order_summary(filtered, ["payment_type"])
    if not type_summary.empty:
        fig = px.bar(type_summary, x="payment_type", y="approval_ratio_%", text="approval_ratio_%", title="Approval Ratio by Payment Type")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        c2.plotly_chart(fig, use_container_width=True)
    st.dataframe(unique_order_summary(filtered, ["payment_type"]), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    if not psp_perf.empty:
        fig = px.bar(psp_perf.sort_values("approval_ratio_%", ascending=True), x="approval_ratio_%", y="psp", orientation="h", text="approval_ratio_%", title="PSP-wise Approval Ratio")
        fig.update_traces(texttemplate="%{text:.2f}%")
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(psp_perf, x="retry_order_ratio_%", y="approval_ratio_%", size="unique_orders", hover_name="psp", title="PSP Approval vs Retry Ratio")
        c2.plotly_chart(fig, use_container_width=True)
    st.dataframe(psp_perf, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    if not country_perf.empty:
        top_countries = country_perf.sort_values("unique_orders", ascending=False).head(25)
        fig = px.bar(top_countries.sort_values("approval_ratio_%", ascending=True), x="approval_ratio_%", y="country", orientation="h", text="approval_ratio_%", title="Top Countries by Volume - Approval Ratio")
        fig.update_traces(texttemplate="%{text:.2f}%")
        c1.plotly_chart(fig, use_container_width=True)
        country_psp = unique_order_summary(filtered, ["country", "psp"])
        top_country_names = top_countries["country"].tolist()
        heat = country_psp[country_psp["country"].isin(top_country_names)]
        if not heat.empty:
            pivot = heat.pivot_table(index="country", columns="psp", values="approval_ratio_%", aggfunc="mean")
            fig = px.imshow(pivot, aspect="auto", title="Country-wise PSP Approval Heatmap")
            c2.plotly_chart(fig, use_container_width=True)
    st.dataframe(country_perf, use_container_width=True)

with tab4:
    declined = filtered[filtered["status_group"] == "Declined"].copy()
    declined["decline_reason_clean"] = declined["decline_reason"].replace({"nan": "Unknown", "": "Unknown"})
    c1, c2 = st.columns(2)
    if not declined.empty:
        top_declines = declined["decline_reason_clean"].value_counts().head(15).reset_index()
        top_declines.columns = ["decline_reason", "attempts"]
        fig = px.bar(top_declines, x="attempts", y="decline_reason", orientation="h", title="Top Decline Reasons")
        c1.plotly_chart(fig, use_container_width=True)

        psp_decline = declined.groupby(["psp", "decline_reason_clean"]).size().reset_index(name="attempts")
        top_reasons = top_declines["decline_reason"].head(10).tolist()
        psp_decline = psp_decline[psp_decline["decline_reason_clean"].isin(top_reasons)]
        if not psp_decline.empty:
            pivot = psp_decline.pivot_table(index="psp", columns="decline_reason_clean", values="attempts", aggfunc="sum", fill_value=0)
            fig = px.imshow(pivot, aspect="auto", title="PSP-to-PSP Decline Reason Comparison")
            c2.plotly_chart(fig, use_container_width=True)

        if declined["txn_date"].notna().any():
            date_decline = declined.groupby(["txn_date", "decline_reason_clean"]).size().reset_index(name="attempts")
            date_decline = date_decline[date_decline["decline_reason_clean"].isin(top_reasons)]
            fig = px.line(date_decline, x="txn_date", y="attempts", color="decline_reason_clean", markers=True, title="Date-wise Decline Reason Comparison")
            st.plotly_chart(fig, use_container_width=True)

        country_decline = declined.groupby(["country", "decline_reason_clean"]).size().reset_index(name="attempts")
        top_country_names = declined["country"].value_counts().head(20).index.tolist()
        country_decline = country_decline[(country_decline["country"].isin(top_country_names)) & (country_decline["decline_reason_clean"].isin(top_reasons))]
        if not country_decline.empty:
            pivot = country_decline.pivot_table(index="country", columns="decline_reason_clean", values="attempts", aggfunc="sum", fill_value=0)
            fig = px.imshow(pivot, aspect="auto", title="Country-wise Decline Reason Comparison")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(psp_decline.sort_values("attempts", ascending=False), use_container_width=True)
    else:
        st.success("No declined attempts found under the current filters.")

with tab5:
    st.markdown("### MID-wise Performance")
    mid_perf = unique_order_summary(filtered, ["mid"])
    if not mid_perf.empty:
        c1, c2 = st.columns(2)
        fig = px.bar(
            mid_perf.sort_values("approval_ratio_%", ascending=True),
            x="approval_ratio_%", y="mid", orientation="h",
            text="approval_ratio_%", title="MID-wise Approval Ratio",
        )
        fig.update_traces(texttemplate="%{text:.2f}%")
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(
            mid_perf, x="unique_orders", y="approval_ratio_%",
            size="total_attempts", hover_name="mid",
            title="MID: Volume vs Approval Ratio",
        )
        c2.plotly_chart(fig, use_container_width=True)
        st.dataframe(mid_perf, use_container_width=True)
    else:
        st.info("No MID data available under the current filters.")

with tab6:
    st.markdown("### Country-wise PSP Routing Recommendation")
    st.caption("Recommendation is based on best observed unique-order approval ratio by country and PSP. Always validate against cost, fraud risk, PSP limits, and compliance rules before changing routing.")
    if routing is not None and not routing.empty:
        st.dataframe(routing, use_container_width=True)
        fig = px.bar(routing.head(25), x="country", y="recommended_approval_%", color="recommended_psp", hover_data=["recommended_unique_orders", "current_country_approval_%"], title="Best PSP by Country")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No routing recommendation found. Reduce the minimum unique orders threshold or adjust filters.")

    st.markdown("### Suggested Actions to Improve Approval Ratio")
    st.markdown(
        """
- Route each country to the PSP with the highest stable approval ratio, but only when the PSP has enough unique order volume.
- Separate retry analysis from approval analysis because repeated retries can make PSP performance look worse at attempt level.
- Review high-decline PSPs by decline reason. If one PSP has more issuer/generic declines in a country, test another PSP/MID for that route.
- Monitor MID-level performance. A weak MID can reduce approval even when the PSP is strong overall.
- For international cards, compare card country and customer country where available to identify cross-border decline pressure.
- Keep Confirmo and PayPal outside card PSP approval benchmarking because their payment behavior is different from international cards.
        """
    )

with tab7:
    st.markdown("### Column Mapping Used")
    st.json({k: v for k, v in mapping.items() if v is not None})
    st.markdown("### Filtered Data")
    st.dataframe(filtered, use_container_width=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data", data=csv, file_name="filtered_orchestrator_data.csv", mime="text/csv")
