import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Orchestrator Approval Dashboard", page_icon="📊", layout="wide")

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
    "date": ["processing_date", "processingDate", "completionDate", "createdAt", "created_at", "date", "transactionDate"],
}


def find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lower_map:
            return lower_map[key]
    for alias in aliases:
        key = alias.strip().lower().replace("_", "").replace(" ", "")
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
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, low_memory=False, encoding_errors="replace")


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mapping = {key: find_col(df, aliases) for key, aliases in COLUMN_ALIASES.items()}
    missing = [k for k in ["psp", "merchant_order_id", "status"] if mapping.get(k) is None]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}. Please check uploaded file headers.")
        st.stop()

    out = pd.DataFrame()
    out["source_row"] = np.arange(1, len(df) + 1)
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


def order_attempts(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    sort_cols = ["merchant_order_id"]
    if data["txn_datetime"].notna().any():
        sort_cols += ["txn_datetime", "source_row"]
    else:
        sort_cols += ["source_row"]
    x = data.sort_values(sort_cols).copy()
    x["attempt_no"] = x.groupby("merchant_order_id").cumcount() + 1
    x["is_first_attempt"] = x["attempt_no"].eq(1)
    x["is_retry_attempt"] = x["attempt_no"].gt(1)
    return x


def unique_order_summary(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    d = order_attempts(data)
    order_level = (
        d.groupby(group_cols + ["merchant_order_id"], dropna=False)
        .agg(
            attempts=("merchant_order_id", "size"),
            approved=("status_group", lambda x: int((x == "Approved").any())),
            first_attempt_approved=("status_group", lambda x: int(x.iloc[0] == "Approved")),
            final_status=("status_group", lambda x: "Approved" if (x == "Approved").any() else x.iloc[-1]),
        )
        .reset_index()
    )
    summary = (
        order_level.groupby(group_cols, dropna=False)
        .agg(
            unique_orders=("merchant_order_id", "nunique"),
            approved_orders=("approved", "sum"),
            first_attempt_approved_orders=("first_attempt_approved", "sum"),
            total_attempts=("attempts", "sum"),
            retried_orders=("attempts", lambda x: int((x > 1).sum())),
            avg_attempts_per_order=("attempts", "mean"),
        )
        .reset_index()
    )
    summary["declined_unique_orders"] = summary["unique_orders"] - summary["approved_orders"]
    summary["approval_ratio_%"] = summary.apply(lambda r: safe_ratio(r["approved_orders"], r["unique_orders"]), axis=1)
    summary["first_attempt_success_rate_%"] = summary.apply(lambda r: safe_ratio(r["first_attempt_approved_orders"], r["unique_orders"]), axis=1)
    summary["retry_order_ratio_%"] = summary.apply(lambda r: safe_ratio(r["retried_orders"], r["unique_orders"]), axis=1)
    summary["retry_attempt_ratio_%"] = summary.apply(lambda r: safe_ratio(r["total_attempts"] - r["unique_orders"], r["total_attempts"]), axis=1)
    summary["approval_lift_after_retry_%"] = summary["approval_ratio_%"] - summary["first_attempt_success_rate_%"]
    return summary.sort_values(["approval_ratio_%", "unique_orders"], ascending=[False, False])


def retry_chain_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = order_attempts(data)
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    chains = []
    transitions = []
    for oid, g in d.groupby("merchant_order_id"):
        g = g.sort_values("attempt_no")
        psps = g["psp"].astype(str).tolist()
        statuses = g["status_group"].astype(str).tolist()
        final_status = "Approved" if "Approved" in statuses else statuses[-1]
        chain = " → ".join(psps[:6]) + (" → ..." if len(psps) > 6 else "")
        chains.append({"merchant_order_id": oid, "chain": chain, "attempts": len(g), "final_status": final_status})
        for i in range(len(psps) - 1):
            transitions.append({"from_psp": psps[i], "to_psp": psps[i + 1], "from_status": statuses[i], "final_order_status": final_status})
    chain_df = pd.DataFrame(chains)
    transition_df = pd.DataFrame(transitions)
    return chain_df, transition_df


def build_routing(data: pd.DataFrame, min_orders: int) -> pd.DataFrame:
    base = unique_order_summary(data, ["country", "psp"])
    if base.empty:
        return base
    base = base[base["unique_orders"] >= min_orders].copy()
    if base.empty:
        return base
    best = base.sort_values(["country", "approval_ratio_%", "first_attempt_success_rate_%", "unique_orders"], ascending=[True, False, False, False])
    best = best.groupby("country", as_index=False).first()
    best = best.rename(columns={"psp": "recommended_psp", "approval_ratio_%": "recommended_approval_%", "first_attempt_success_rate_%": "recommended_fasr_%", "unique_orders": "recommended_unique_orders"})
    country_total = unique_order_summary(data, ["country"])[["country", "unique_orders", "approved_orders", "approval_ratio_%", "first_attempt_success_rate_%"]]
    country_total = country_total.rename(columns={"approval_ratio_%": "current_country_approval_%", "first_attempt_success_rate_%": "current_country_fasr_%"})
    rec = best.merge(country_total, on="country", how="left", suffixes=("", "_country"))
    rec["potential_approval_gap_%"] = rec["recommended_approval_%"] - rec["current_country_approval_%"]
    rec["action_priority"] = np.select(
        [rec["potential_approval_gap_%"] >= 10, rec["potential_approval_gap_%"] >= 5, rec["potential_approval_gap_%"] > 0],
        ["High", "Medium", "Low"], default="Monitor"
    )
    rec["routing_insight"] = np.where(
        rec["potential_approval_gap_%"] > 0,
        "Shift more traffic to the recommended PSP for this country after checking cost, risk and limits.",
        "Current country mix is close to the best observed route."
    )
    keep = ["country", "recommended_psp", "action_priority", "recommended_approval_%", "recommended_fasr_%", "recommended_unique_orders", "current_country_approval_%", "current_country_fasr_%", "unique_orders", "potential_approval_gap_%", "retry_order_ratio_%", "avg_attempts_per_order", "routing_insight"]
    return rec[[c for c in keep if c in rec.columns]].sort_values(["action_priority", "potential_approval_gap_%", "recommended_unique_orders"], ascending=[True, False, False])


st.title("📊 Orchestrator Approval, Retry & Routing Dashboard v2")
st.caption("Unique Merchant Order ID logic | Confirmo = Crypto | PayPal = P2P | All other PSPs = International Card")

with st.sidebar:
    st.header("Upload & Filters")
    uploaded = st.file_uploader("Upload orchestrator CSV", type=["csv"])
    min_orders = st.number_input("Minimum unique orders for routing recommendation", min_value=1, value=10, step=1)

if uploaded is None:
    st.info("Upload your orchestrator CSV report to start the dashboard.")
    st.stop()

raw = load_csv(uploaded)
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
if payment_types:
    filtered = filtered[filtered["payment_type"].isin(payment_types)]
if countries:
    filtered = filtered[filtered["country"].isin(countries)]
if psps:
    filtered = filtered[filtered["psp"].isin(psps)]
if mids:
    filtered = filtered[filtered["mid"].isin(mids)]

attempted = order_attempts(filtered)
order_level = attempted.groupby("merchant_order_id").agg(attempts=("merchant_order_id", "size"), approved=("status_group", lambda x: int((x == "Approved").any())), first_approved=("status_group", lambda x: int(x.iloc[0] == "Approved"))).reset_index() if not attempted.empty else pd.DataFrame()
unique_orders = int(order_level["merchant_order_id"].nunique()) if not order_level.empty else 0
approved_orders = int(order_level["approved"].sum()) if not order_level.empty else 0
first_approved = int(order_level["first_approved"].sum()) if not order_level.empty else 0
total_attempts = int(len(filtered))
retried_orders = int((order_level["attempts"] > 1).sum()) if not order_level.empty else 0
approval_ratio = safe_ratio(approved_orders, unique_orders)
fasr = safe_ratio(first_approved, unique_orders)
retry_order_ratio = safe_ratio(retried_orders, unique_orders)
retry_attempt_ratio = safe_ratio(total_attempts - unique_orders, total_attempts)
retry_lift = approval_ratio - fasr

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Unique Orders", f"{unique_orders:,}")
k2.metric("Approval Ratio", f"{approval_ratio:.2f}%")
k3.metric("First Attempt Success", f"{fasr:.2f}%")
k4.metric("Approval Lift from Retry", f"{retry_lift:.2f}%")
k5.metric("Retry Order Ratio", f"{retry_order_ratio:.2f}%")
k6.metric("Retry Attempt Ratio", f"{retry_attempt_ratio:.2f}%")

st.divider()
psp_perf = unique_order_summary(filtered, ["psp"])
country_perf = unique_order_summary(filtered, ["country"])
routing = build_routing(filtered, int(min_orders))
chain_df, transition_df = retry_chain_summary(filtered)

st.subheader("Executive Updates")
updates = []
if not psp_perf.empty:
    best = psp_perf.iloc[0]
    worst_pool = psp_perf[psp_perf["unique_orders"] >= max(3, int(min_orders))]
    worst = worst_pool.sort_values("approval_ratio_%").iloc[0] if not worst_pool.empty else psp_perf.sort_values("approval_ratio_%").iloc[0]
    updates.append(f"Best PSP: **{best['psp']}** with **{best['approval_ratio_%']:.2f}%** unique-order approval and **{best['first_attempt_success_rate_%']:.2f}%** first-attempt success.")
    updates.append(f"PSP needing review: **{worst['psp']}** with **{worst['approval_ratio_%']:.2f}%** approval from **{int(worst['unique_orders']):,}** unique orders.")
if retry_lift > 5:
    updates.append(f"Retries are materially improving final approval by **{retry_lift:.2f}%**, but they also create customer friction. Optimize first-attempt routing first.")
if retry_attempt_ratio > 35:
    updates.append(f"Retry attempt pressure is high at **{retry_attempt_ratio:.2f}%**. Review repeated PSP loops and switch PSP/MID after defined failure rules.")
if routing is not None and not routing.empty:
    top_route = routing.sort_values("potential_approval_gap_%", ascending=False).iloc[0]
    updates.append(f"Top country routing opportunity: **{top_route['country']} → {top_route['recommended_psp']}** with estimated approval gap of **{top_route['potential_approval_gap_%']:.2f}%**.")
for u in updates:
    st.markdown(f"- {u}")

st.divider()
tabs = st.tabs(["Overview", "First Attempt & Retry", "PSP Analysis", "Country Analysis", "Decline Reasons", "Routing Insights", "Raw Data"])

with tabs[0]:
    c1, c2 = st.columns(2)
    daily = unique_order_summary(filtered.dropna(subset=["txn_date"]), ["txn_date"])
    if not daily.empty:
        fig = px.line(daily, x="txn_date", y=["approval_ratio_%", "first_attempt_success_rate_%"], markers=True, title="Date-wise Approval vs First Attempt Success")
        c1.plotly_chart(fig, use_container_width=True)
    type_summary = unique_order_summary(filtered, ["payment_type"])
    if not type_summary.empty:
        fig = px.bar(type_summary, x="payment_type", y="approval_ratio_%", text="approval_ratio_%", title="Approval Ratio by Payment Type")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        c2.plotly_chart(fig, use_container_width=True)
    st.dataframe(type_summary, use_container_width=True)

with tabs[1]:
    c1, c2 = st.columns(2)
    if not psp_perf.empty:
        fig = px.scatter(psp_perf, x="first_attempt_success_rate_%", y="approval_ratio_%", size="unique_orders", hover_name="psp", title="First Attempt Success vs Final Approval by PSP")
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.bar(psp_perf.sort_values("approval_lift_after_retry_%", ascending=True), x="approval_lift_after_retry_%", y="psp", orientation="h", text="approval_lift_after_retry_%", title="Approval Lift After Retry by PSP")
        fig.update_traces(texttemplate="%{text:.2f}%")
        c2.plotly_chart(fig, use_container_width=True)
    if not chain_df.empty:
        st.markdown("### Top Retry Chains")
        top_chains = chain_df[chain_df["attempts"] > 1].groupby(["chain", "final_status"]).agg(unique_orders=("merchant_order_id", "nunique"), avg_attempts=("attempts", "mean")).reset_index().sort_values("unique_orders", ascending=False).head(25)
        st.dataframe(top_chains, use_container_width=True)
    if not transition_df.empty:
        st.markdown("### PSP-to-PSP Retry Transition Flow")
        flows = transition_df.groupby(["from_psp", "to_psp", "final_order_status"]).size().reset_index(name="orders")
        st.dataframe(flows.sort_values("orders", ascending=False), use_container_width=True)

with tabs[2]:
    c1, c2 = st.columns(2)
    if not psp_perf.empty:
        fig = px.bar(psp_perf.sort_values("approval_ratio_%", ascending=True), x="approval_ratio_%", y="psp", orientation="h", text="approval_ratio_%", title="PSP-wise Approval Ratio")
        fig.update_traces(texttemplate="%{text:.2f}%")
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(psp_perf, x="retry_order_ratio_%", y="approval_ratio_%", size="unique_orders", hover_name="psp", title="PSP Approval vs Retry Ratio")
        c2.plotly_chart(fig, use_container_width=True)
    st.dataframe(psp_perf, use_container_width=True)

with tabs[3]:
    c1, c2 = st.columns(2)
    if not country_perf.empty:
        top_countries = country_perf.sort_values("unique_orders", ascending=False).head(25)
        fig = px.bar(top_countries.sort_values("approval_ratio_%", ascending=True), x="approval_ratio_%", y="country", orientation="h", text="approval_ratio_%", title="Top Countries by Volume - Approval Ratio")
        fig.update_traces(texttemplate="%{text:.2f}%")
        c1.plotly_chart(fig, use_container_width=True)
        country_psp = unique_order_summary(filtered, ["country", "psp"])
        heat = country_psp[country_psp["country"].isin(top_countries["country"].tolist())]
        if not heat.empty:
            pivot = heat.pivot_table(index="country", columns="psp", values="approval_ratio_%", aggfunc="mean")
            fig = px.imshow(pivot, aspect="auto", title="Country-wise PSP Approval Heatmap")
            c2.plotly_chart(fig, use_container_width=True)
    st.dataframe(country_perf, use_container_width=True)

with tabs[4]:
    declined = filtered[filtered["status_group"] != "Approved"].copy()
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

with tabs[5]:
    st.markdown("### Country-wise PSP Routing Recommendation")
    st.caption("Based on best observed unique-order approval ratio by country and PSP. Validate cost, fraud risk, PSP limits and compliance before changing routing.")
    if routing is not None and not routing.empty:
        st.dataframe(routing, use_container_width=True)
        fig = px.bar(routing.sort_values("potential_approval_gap_%", ascending=False).head(25), x="country", y="potential_approval_gap_%", color="recommended_psp", hover_data=["recommended_approval_%", "current_country_approval_%", "recommended_unique_orders"], title="Potential Approval Gap by Recommended Route")
        st.plotly_chart(fig, use_container_width=True)
        csv = routing.to_csv(index=False).encode("utf-8")
        st.download_button("Download routing recommendations", data=csv, file_name="country_psp_routing_recommendations.csv", mime="text/csv")
    else:
        st.warning("No routing recommendation found. Reduce the minimum unique orders threshold or adjust filters.")
    st.markdown("### Practical Optimization Rules")
    st.markdown("""
- Use **First Attempt Success Rate** as the main routing KPI. Final approval can hide too many customer retries.
- If one PSP fails twice for the same order, route the next attempt to a different PSP or MID.
- Keep **Confirmo** and **PayPal** separate from international-card benchmarking.
- For each country, push more volume to PSPs with stable approval and enough unique-order volume.
- Investigate PSPs where retry lift is high but first-attempt success is low. This usually means the route works only after customer friction.
- Compare decline reasons by PSP before changing routing. A high “generic/issuer decline” concentration may require a different acquiring route.
""")

with tabs[6]:
    st.markdown("### Column Mapping Used")
    st.json(mapping)
    st.markdown("### Filtered Data with Attempt Number")
    st.dataframe(attempted, use_container_width=True)
    csv = attempted.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data", data=csv, file_name="filtered_orchestrator_data_with_attempts.csv", mime="text/csv")
