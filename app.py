import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Customer Segmentation Studio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Professional CSS Styling (Glassmorphism & 3D Cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stApp {
        background: #090d16;
        color: #e2e8f0;
    }
    
    /* Executive Header */
    .hero-container {
        padding: 24px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    /* 3D Dynamic Metric Cards with Lift Effect */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 18px 22px;
        border-radius: 14px;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.01);
        border-color: #818cf8;
        box-shadow: 0px 12px 24px rgba(129, 140, 248, 0.25);
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricValue"] div {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* Tab Layout Tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. App Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">💎 Enterprise Customer Intelligence Platform</div>
    <div style="color: #94a3b8; font-size: 0.95rem;">Automated Behavioral & Demographic Customer Segmentation Engine</div>
</div>
""", unsafe_allow_html=True)

# 4. Data Loader (Supports Uploads + Default Fallback)
@st.cache_data
def get_default_data():
    np.random.seed(42)
    n = 500
    data = {
        "CustomerID": [f"CUST-{1000+i}" for i in range(n)],
        "Age": np.random.randint(18, 70, size=n),
        "Annual_Income_k$": np.random.randint(15, 150, size=n),
        "Spending_Score": np.random.randint(1, 100, size=n),
        "Purchase_Frequency": np.random.randint(1, 60, size=n),
        "Total_Spent": np.random.randint(200, 12000, size=n)
    }
    return pd.DataFrame(data)

st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Customer Dataset (CSV):", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("customer_data.csv")
    except Exception:
        df = get_default_data()

# 5. Sidebar Model Parameters
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Tuning")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "CustomerID" in numeric_cols:
    numeric_cols.remove("CustomerID")

default_selected = [c for c in ["Annual_Income_k$", "Spending_Score", "Total_Spent"] if c in numeric_cols]

selected_features = st.sidebar.multiselect(
    "Select Features for Clustering:",
    options=numeric_cols,
    default=default_selected if len(default_selected) >= 2 else numeric_cols[:2]
)

if len(selected_features) < 2:
    st.warning("⚠️ Select at least 2 features in the sidebar to execute segmentation.")
    st.stop()

k_clusters = st.sidebar.slider("Number of Segments (K):", min_value=2, max_value=8, value=4)

# 6. ML Execution & Metrics (Safe handling of missing records)
df = df.dropna(subset=selected_features).reset_index(drop=True)
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df["Segment"] = kmeans.fit_predict(X_scaled)
df["Segment_Label"] = "Segment " + df["Segment"].astype(str)

sil_score = silhouette_score(X_scaled, kmeans.labels_)

# Chart Styling Helper
def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        margin=dict(l=10, r=10, t=35, b=10)
    )
    return fig

# 7. Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "📈 3D & Cluster Analytics", 
    "🧠 Model Evaluation (Elbow)", 
    "🔍 Export & Profile Explorer"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{len(df):,}")
    m2.metric("Active Segments", k_clusters)
    m3.metric("Silhouette Score", f"{sil_score:.2f}")
    m4.metric("Avg Annual Income", f"${df['Annual_Income_k$'].mean():.1f}k" if "Annual_Income_k$" in df else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("2D Feature Relationship Scatter")
        fig_2d = px.scatter(
            df, x=selected_features[0], y=selected_features[1],
            color="Segment_Label",
            size=selected_features[2] if len(selected_features) > 2 else None,
            hover_data=["CustomerID"],
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(apply_dark_theme(fig_2d), use_container_width=True)
        
    with col2:
        st.subheader("Customer Distribution")
        fig_pie = px.pie(
            df, names="Segment_Label", hole=0.45,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(apply_dark_theme(fig_pie), use_container_width=True)

# --- TAB 2: 3D & CLUSTER ANALYTICS ---
with tab2:
    if len(selected_features) >= 3:
        st.subheader("3D Interactive Cluster Projection")
        fig_3d = px.scatter_3d(
            df, 
            x=selected_features[0], 
            y=selected_features[1], 
            z=selected_features[2],
            color="Segment_Label",
            hover_data=["CustomerID"],
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=600
        )
        st.plotly_chart(apply_dark_theme(fig_3d), use_container_width=True)
    else:
        st.info("💡 Select at least 3 features in the sidebar to unlock 3D Cluster Visualizations.")

    st.subheader("Segment Means Breakdown")
    profile = df.groupby("Segment_Label")[selected_features].mean().reset_index()
    st.dataframe(
        profile.style.format({col: "{:.1f}" for col in selected_features})
        .background_gradient(cmap="Blues", subset=selected_features),
        use_container_width=True
    )

# --- TAB 3: MODEL EVALUATION (ELBOW METHOD) ---
with tab3:
    st.subheader("Optimal Cluster Analysis (Elbow Method & Inertia)")
    st.write("This curve measures how the sum of squared distances decreases as $K$ increases. The 'elbow' point indicates optimal cluster count.")
    
    @st.cache_data
    def compute_elbow(data_scaled):
        inertias = []
        K_range = range(2, 10)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(data_scaled)
            inertias.append(km.inertia_)
        return list(K_range), inertias

    k_vals, inertias = compute_elbow(X_scaled)
    
    fig_elbow = px.line(
        x=k_vals, y=inertias, markers=True,
        labels={"x": "Number of Clusters (K)", "y": "Inertia (Within-Cluster Distance)"},
        template="plotly_dark"
    )
    fig_elbow.add_vline(x=k_clusters, line_dash="dash", line_color="#ef4444", annotation_text=f"Selected K={k_clusters}")
    st.plotly_chart(apply_dark_theme(fig_elbow), use_container_width=True)

# --- TAB 4: EXPORT & LOOKUP ---
with tab4:
    st.subheader("Customer Profiles Explorer")
    
    c_search, c_filter = st.columns([2, 1])
    with c_search:
        cust_id = st.selectbox("Lookup Customer ID:", options=df["CustomerID"].unique())
    with c_filter:
        seg_filter = st.selectbox("Filter Table by Segment:", options=["All"] + list(df["Segment_Label"].unique()))

    # Individual Card
    cust_data = df[df["CustomerID"] == cust_id].iloc[0]
    st.markdown(f"#### Customer Details: **{cust_id}** (`{cust_data['Segment_Label']}`)")
    
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Income", f"${cust_data.get('Annual_Income_k$', 'N/A')}k")
    ic2.metric("Spending Score", f"{cust_data.get('Spending_Score', 'N/A')} / 100")
    ic3.metric("Total Spent", f"${cust_data.get('Total_Spent', 'N/A'):,}")
    
    st.markdown("---")
    
    # Export Data Section
    filtered_df = df if seg_filter == "All" else df[df["Segment_Label"] == seg_filter]
    
    st.subheader("Segment Dataset")
    st.dataframe(filtered_df[["CustomerID", "Segment_Label"] + selected_features], use_container_width=True)
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Segmented Data as CSV",
        data=csv_data,
        file_name="customer_segmentation_results.csv",
        mime="text/csv"
    )