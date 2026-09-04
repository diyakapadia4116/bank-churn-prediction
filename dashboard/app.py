import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap

from utils import (
    load_pipeline, load_metadata, load_leaderboard, load_feature_importance,
    load_population_scores, load_raw_data, build_feature_row, risk_tier
)

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Token system: light blue + white, glass/gradient modern fintech look ---
INK = "#12233D"          # body text — deep blue-gray, never pure black
NAVY = "#1E5C99"          # heading accent (kept name for compatibility below)
BLUE = "#4A9FE0"          # primary accent
MID_BLUE = "#6DB4E8"
LIGHT_BLUE = "#9AD0F5"
SKY = "#D9ECFB"           # soft blue surface
ICE = "#F3F9FF"           # near-white page wash
WHITE = "#FFFFFF"
RED = "#E0625F"
GREEN = "#2FAE79"
AMBER = "#E8A13D"
SEQ_BLUES = ["#1E5C99", "#4A9FE0", "#6DB4E8", "#9AD0F5", "#D9ECFB"]

st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, {ICE} 0%, {WHITE} 55%);
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(200deg, #DDEEFC 0%, #F6FBFF 55%, {WHITE} 100%);
        border-right: 1px solid #E1EEF9;
    }}
    section[data-testid="stSidebar"] * {{
        color: {INK} !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: #D3E7F8;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        background: {WHITE};
        border: 1px solid #DCEBFA;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 6px;
        transition: all 0.15s ease;
        box-shadow: 0 1px 3px rgba(30,92,153,0.05);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: linear-gradient(90deg, {SKY}, {WHITE});
        border-color: {BLUE};
    }}
    h1, h2, h3, h4 {{
        color: {NAVY};
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
    }}
    h1 {{
        background: linear-gradient(90deg, {NAVY}, {BLUE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
    }}
    p, span, label, div {{
        color: {INK};
    }}
    .kpi-card {{
        background: linear-gradient(135deg, {BLUE} 0%, {LIGHT_BLUE} 100%);
        border-radius: 16px;
        padding: 20px 22px;
        color: white !important;
        box-shadow: 0 8px 24px rgba(74,159,224,0.28);
        text-align: left;
        border: 1px solid rgba(255,255,255,0.4);
        position: relative;
        overflow: hidden;
    }}
    .kpi-card::after {{
        content: "";
        position: absolute;
        top: -40%; right: -20%;
        width: 140px; height: 140px;
        background: radial-gradient(circle, rgba(255,255,255,0.35) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }}
    .kpi-card * {{ color: white !important; }}
    .kpi-label {{
        font-size: 13px;
        opacity: 0.92;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 30px;
        font-weight: 800;
        margin-top: 2px;
    }}
    .section-card {{
        background: linear-gradient(135deg, {ICE} 0%, {WHITE} 100%);
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid #E1EEF9;
        box-shadow: 0 4px 16px rgba(74,159,224,0.08);
        margin-bottom: 14px;
    }}
    .risk-badge {{
        display: inline-block;
        padding: 7px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 15px;
        color: white !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    .divider-blue {{
        height: 4px;
        background: linear-gradient(90deg, {BLUE}, {LIGHT_BLUE}, {WHITE});
        border-radius: 4px;
        margin: 6px 0 20px 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {WHITE};
        border: 1px solid #E1EEF9;
        border-radius: 10px 10px 0 0;
        color: {NAVY};
        font-weight: 600;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {BLUE}, {MID_BLUE}) !important;
        color: white !important;
        border-color: transparent;
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, {WHITE} 0%, {SKY} 100%);
        border: 1px solid #E1EEF9;
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 4px 14px rgba(74,159,224,0.10);
    }}
    div[data-testid="stMetricValue"] {{
        color: {NAVY};
    }}
    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(90deg, {BLUE}, {MID_BLUE});
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(74,159,224,0.25);
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background: {BLUE};
    }}
    div[data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E1EEF9;
    }}
    hr {{
        border-color: #E1EEF9;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
pipeline = load_pipeline()
metadata = load_metadata()
leaderboard = load_leaderboard()
fi_df = load_feature_importance()
population_scores = load_population_scores()
raw_df = load_raw_data()

FEATURE_COLUMNS = metadata["feature_columns"]
GEO_OPTIONS = metadata["geography_options"]
GENDER_OPTIONS = metadata["gender_options"]
RANGES = metadata["feature_ranges"]
BEST_MODEL = metadata["best_model_name"]

merged = population_scores.merge(
    raw_df[["CustomerId", "Geography", "Gender", "IsActiveMember", "NumOfProducts", "Age"]],
    on="CustomerId", how="left"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 Churn Intelligence")
    st.caption("European Central Bank · Retail Customer Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Churn Risk Calculator", "What-If Simulator",
         "Population Analytics", "Feature Importance", "Model Performance"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**Active model:** {BEST_MODEL}")
    st.markdown(f"**Test ROC-AUC:** {metadata['test_metrics']['ROC-AUC']:.3f}")
    st.caption("Model trained on 10,000 retail banking customers across France, Germany & Spain.")

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Customer Churn Risk Intelligence")
    st.markdown("Predictive churn intelligence system — risk scoring, driver explainability, and retention scenario planning.")
    st.markdown('<div class="divider-blue"></div>', unsafe_allow_html=True)

    total_customers = len(raw_df)
    churn_rate = raw_df["Exited"].mean()
    active_rate = raw_df["IsActiveMember"].mean()
    avg_balance = raw_df["Balance"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Total Customers", "Overall Churn Rate", "Active Members", "Avg. Account Balance"],
        [f"{total_customers:,}", f"{churn_rate:.1%}", f"{active_rate:.1%}", f"€{avg_balance:,.0f}"]
    ):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("Churn Composition")
        fig = px.pie(
            raw_df, names=raw_df["Exited"].map({0: "Retained", 1: "Churned"}),
            color=raw_df["Exited"].map({0: "Retained", 1: "Churned"}),
            color_discrete_map={"Retained": BLUE, "Churned": RED},
            hole=0.55,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor=WHITE, plot_bgcolor=WHITE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Geography & Gender")
        seg = raw_df.groupby(["Geography", "Gender"])["Exited"].mean().reset_index()
        seg["Exited"] = seg["Exited"] * 100
        fig = px.bar(
            seg, x="Geography", y="Exited", color="Gender", barmode="group",
            color_discrete_sequence=[BLUE, MID_BLUE],
            labels={"Exited": "Churn Rate (%)"},
        )
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="section-card">
    <b>Key business insight:</b> Inactive members and customers holding 3–4 products churn at a
    substantially higher rate than the base population, and Germany runs a structurally higher churn
    rate than France or Spain. See <b>Feature Importance</b> for the full explainability breakdown, and
    use the <b>Churn Risk Calculator</b> to score an individual customer.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: Churn Risk Calculator
# ---------------------------------------------------------------------------
elif page == "Churn Risk Calculator":
    st.title("Churn Risk Calculator")
    st.caption("Enter a customer's profile to generate a live churn probability score.")
    st.markdown('<div class="divider-blue"></div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("#### Customer Profile")
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox("Geography", GEO_OPTIONS)
            age = st.slider("Age", RANGES["Age"][0], RANGES["Age"][1], 40)
            credit_score = st.slider("Credit Score", RANGES["CreditScore"][0], RANGES["CreditScore"][1], 650)
            tenure = st.slider("Tenure (years)", RANGES["Tenure"][0], RANGES["Tenure"][1], 5)
            has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        with c2:
            gender = st.selectbox("Gender", GENDER_OPTIONS)
            balance = st.number_input("Account Balance (€)", min_value=0.0,
                                       max_value=float(RANGES["Balance"][1]), value=75000.0, step=1000.0)
            salary = st.number_input("Estimated Salary (€)", min_value=float(RANGES["EstimatedSalary"][0]),
                                      max_value=float(RANGES["EstimatedSalary"][1]), value=100000.0, step=1000.0)
            num_products = st.slider("Number of Products", RANGES["NumOfProducts"][0], RANGES["NumOfProducts"][1], 2)
            is_active = st.selectbox("Active Member", ["Yes", "No"])

        row = build_feature_row(
            CreditScore=credit_score, Geography=geography, Gender=gender, Age=age,
            Tenure=tenure, Balance=balance, NumOfProducts=num_products,
            HasCrCard=1 if has_cr_card == "Yes" else 0,
            IsActiveMember=1 if is_active == "Yes" else 0,
            EstimatedSalary=salary, feature_columns=FEATURE_COLUMNS
        )
        proba = pipeline.predict_proba(row)[0, 1]
        tier, tier_color = risk_tier(proba)

    with right:
        st.markdown("#### Churn Probability")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 42, "color": NAVY}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": NAVY},
                "bar": {"color": tier_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "#E4F3EA"},
                    {"range": [30, 60], "color": "#FBF0DC"},
                    {"range": [60, 100], "color": "#F8E3E3"},
                ],
                "threshold": {"line": {"color": NAVY, "width": 3}, "thickness": 0.8, "value": proba * 100},
            }
        ))
        fig.update_layout(height=300, margin=dict(t=20, b=10, l=30, r=30), paper_bgcolor=WHITE)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div style="text-align:center;"><span class="risk-badge" style="background-color:{tier_color};">{tier}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Top Factors for This Customer")
        model = pipeline.named_steps["model"]
        preproc = pipeline.named_steps["preprocessor"]
        row_transformed = preproc.transform(row)
        feature_names = preproc.get_feature_names_out()

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(row_transformed)
        shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        contrib = pd.DataFrame({
            "feature": feature_names,
            "impact": shap_vals[0]
        }).sort_values("impact", key=abs, ascending=False).head(6)
        contrib["direction"] = np.where(contrib["impact"] > 0, "Increases risk", "Reduces risk")

        fig2 = px.bar(
            contrib.sort_values("impact"), x="impact", y="feature", orientation="h",
            color="direction", color_discrete_map={"Increases risk": RED, "Reduces risk": GREEN},
        )
        fig2.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, height=280,
                            margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: What-If Simulator
# ---------------------------------------------------------------------------
elif page == "What-If Simulator":
    st.title("What-If Scenario Simulator")
    st.caption("Adjust engagement and product variables to see how churn probability responds — for testing retention interventions.")
    st.markdown('<div class="divider-blue"></div>', unsafe_allow_html=True)

    st.markdown("#### Baseline Customer")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        geography = st.selectbox("Geography", GEO_OPTIONS, key="wi_geo")
    with b2:
        gender = st.selectbox("Gender", GENDER_OPTIONS, key="wi_gender")
    with b3:
        age = st.slider("Age", RANGES["Age"][0], RANGES["Age"][1], 45, key="wi_age")
    with b4:
        credit_score = st.slider("Credit Score", RANGES["CreditScore"][0], RANGES["CreditScore"][1], 620, key="wi_cs")
    with b5:
        tenure = st.slider("Tenure", RANGES["Tenure"][0], RANGES["Tenure"][1], 3, key="wi_tenure")

    b6, b7 = st.columns(2)
    with b6:
        balance = st.number_input("Balance (€)", min_value=0.0, max_value=float(RANGES["Balance"][1]),
                                   value=120000.0, step=1000.0, key="wi_balance")
    with b7:
        salary = st.number_input("Salary (€)", min_value=float(RANGES["EstimatedSalary"][0]),
                                  max_value=float(RANGES["EstimatedSalary"][1]), value=80000.0, step=1000.0, key="wi_salary")

    base_products = 3
    base_active = 0
    base_card = 1

    base_row = build_feature_row(
        CreditScore=credit_score, Geography=geography, Gender=gender, Age=age, Tenure=tenure,
        Balance=balance, NumOfProducts=base_products, HasCrCard=base_card,
        IsActiveMember=base_active, EstimatedSalary=salary, feature_columns=FEATURE_COLUMNS
    )
    base_proba = pipeline.predict_proba(base_row)[0, 1]

    st.markdown("#### Retention Levers")
    l1, l2, l3 = st.columns(3)
    with l1:
        sim_active = st.toggle("Convert to Active Member", value=False)
    with l2:
        sim_products = st.slider("Adjust Number of Products", RANGES["NumOfProducts"][0], RANGES["NumOfProducts"][1], base_products)
    with l3:
        sim_card = st.toggle("Has Credit Card", value=bool(base_card))

    sim_row = build_feature_row(
        CreditScore=credit_score, Geography=geography, Gender=gender, Age=age, Tenure=tenure,
        Balance=balance, NumOfProducts=sim_products, HasCrCard=1 if sim_card else 0,
        IsActiveMember=1 if sim_active else base_active, EstimatedSalary=salary,
        feature_columns=FEATURE_COLUMNS
    )
    sim_proba = pipeline.predict_proba(sim_row)[0, 1]
    delta = sim_proba - base_proba

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Churn Probability", f"{base_proba:.1%}")
    m2.metric("Scenario Churn Probability", f"{sim_proba:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
    tier, tier_color = risk_tier(sim_proba)
    m3.markdown(f"""
        <div style="padding-top:8px;">
        <span class="risk-badge" style="background-color:{tier_color};">{tier}</span>
        </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Baseline", "Scenario"], y=[base_proba * 100, sim_proba * 100],
                          marker_color=[LIGHT_BLUE, BLUE], text=[f"{base_proba:.1%}", f"{sim_proba:.1%}"],
                          textposition="outside"))
    fig.update_layout(yaxis_title="Churn Probability (%)", paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                       height=350, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="section-card">
    Moving this customer from <b>inactive, {base_products} products</b> to the scenario configuration changes
    churn probability by <b>{delta:+.1%}</b>. Use this simulator to stress-test retention offers
    (re-engagement campaigns, product consolidation, card issuance) before committing budget to a segment.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: Population Analytics
# ---------------------------------------------------------------------------
elif page == "Population Analytics":
    st.title("Population Risk Analytics")
    st.caption("Churn probability distribution across the full scored customer base.")
    st.markdown('<div class="divider-blue"></div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        geo_filter = st.multiselect("Geography", GEO_OPTIONS, default=GEO_OPTIONS)
    with f2:
        gender_filter = st.multiselect("Gender", GENDER_OPTIONS, default=GENDER_OPTIONS)
    with f3:
        active_filter = st.multiselect("Activity Status", ["Active", "Inactive"], default=["Active", "Inactive"])

    active_map = {"Active": 1, "Inactive": 0}
    active_vals = [active_map[a] for a in active_filter]

    filtered = merged[
        merged["Geography"].isin(geo_filter) &
        merged["Gender"].isin(gender_filter) &
        merged["IsActiveMember"].isin(active_vals)
    ]

    k1, k2, k3 = st.columns(3)
    k1.metric("Customers in Segment", f"{len(filtered):,}")
    k2.metric("Avg. Predicted Churn Probability", f"{filtered['ChurnProbability'].mean():.1%}")
    k3.metric("High-Risk Customers (≥60%)", f"{(filtered['ChurnProbability'] >= 0.6).sum():,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(filtered, x="ChurnProbability", nbins=40,
                            color_discrete_sequence=[BLUE])
        fig.add_vline(x=0.5, line_dash="dash", line_color=RED, annotation_text="Decision threshold")
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                           xaxis_title="Predicted Churn Probability", yaxis_title="Customer Count",
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Risk by Product Count")
        prod_seg = filtered.groupby("NumOfProducts")["ChurnProbability"].mean().reset_index()
        fig = px.bar(prod_seg, x="NumOfProducts", y="ChurnProbability",
                     color_discrete_sequence=[BLUE])
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                           yaxis_title="Avg. Churn Probability", xaxis_title="Number of Products",
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 Highest-Risk Customers in Segment")
    top_risk = filtered.sort_values("ChurnProbability", ascending=False).head(20)[
        ["CustomerId", "Geography", "Gender", "Age", "NumOfProducts", "IsActiveMember", "ChurnProbability"]
    ]
    top_risk["ChurnProbability"] = (top_risk["ChurnProbability"] * 100).round(1).astype(str) + "%"
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: Feature Importance
# ---------------------------------------------------------------------------
elif page == "Feature Importance":
    st.title("Model Explainability")
    st.caption(f"What drives churn predictions for the {BEST_MODEL} model.")
    st.markdown('<div class="divider-blue"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Feature Importance Ranking")
        top_n = st.slider("Show top N features", 5, len(fi_df), 12)
        fi_top = fi_df.head(top_n).sort_values("importance")
        fig = px.bar(fi_top, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=[SKY, BLUE, NAVY])
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, coloraxis_showscale=False,
                           height=max(350, 28 * top_n), margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("SHAP Global Impact")
        st.image("models/shap_summary.png", use_container_width=True)
        st.caption("Each point is a customer; red = higher feature value, blue = lower. Position shows impact on predicted churn.")

    st.subheader("Partial Dependence — Top Drivers")
    st.image("models/partial_dependence.png", use_container_width=True)
    st.caption("How the average predicted churn probability shifts as each top feature changes, holding others constant.")

    st.markdown("""
    <div class="section-card">
    <b>Explainability takeaways:</b> Member activity status, number of products held, age, and geography
    (Germany) are the dominant churn drivers. Product count has a non-linear effect — 3–4 products
    correlates with <i>higher</i> churn, likely reflecting failed cross-sell or dissatisfaction rather
    than loyalty.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance & Benchmarking")
    st.caption("Comparison across all candidate models trained in the research notebook.")
    st.markdown('<div class="divider-blue"></div>', unsafe_allow_html=True)

    st.subheader("Model Leaderboard")
    lb_display = leaderboard.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        lb_display[c] = (lb_display[c] * 100).round(2)
    st.dataframe(
        lb_display.style.background_gradient(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                                               cmap="Blues"),
        use_container_width=True, hide_index=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve Comparison")
        st.image("models/roc_comparison.png", use_container_width=True)
    with col2:
        st.subheader(f"Confusion Matrix — {BEST_MODEL}")
        st.image("models/confusion_matrix.png", use_container_width=True)

    st.subheader("Precision / Recall vs. Decision Threshold")
    st.image("models/threshold_tradeoff.png", use_container_width=True)
    st.caption("The default 0.5 threshold can be moved to trade off false alarms against missed churners, depending on retention campaign cost.")

    m = metadata["test_metrics"]
    st.markdown(f"""
    <div class="section-card">
    <b>Selected model:</b> {BEST_MODEL} &nbsp;|&nbsp;
    <b>Accuracy:</b> {m['Accuracy']:.1%} &nbsp;|&nbsp;
    <b>Precision:</b> {m['Precision']:.1%} &nbsp;|&nbsp;
    <b>Recall:</b> {m['Recall']:.1%} &nbsp;|&nbsp;
    <b>F1:</b> {m['F1-Score']:.1%} &nbsp;|&nbsp;
    <b>ROC-AUC:</b> {m['ROC-AUC']:.3f}
    </div>
    """, unsafe_allow_html=True)
