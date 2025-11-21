# app_simple.py
# Streamlit interface for Logistic Regression Sentiment Analysis
# Features: single review, batch CSV, bar + pie charts, export PDF report with charts
# Run with: streamlit run app_simple.py

import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ===== CONFIG =====
MODEL_PATH = "model/logistic_regression_model.pkl"
VECT_PATH = "model/tfidf_vectorizer.pkl"

# ===== ENHANCED PAGE CONFIG =====
st.set_page_config(
    page_title="Sentiment Analyzer Pro",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS FOR ENHANCED STYLING =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem !important;
    }
    
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 2rem !important;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3);
    }
    
    .success-box {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #4c51bf 0%, #2d3748 100%);
        color: white;
    }
    
    .uploaded-file {
        background: #f7fafc;
        border: 2px dashed #cbd5e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== LOAD MODEL & VECTORIZER =====
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_vectorizer():
    if not os.path.exists(VECT_PATH):
        raise FileNotFoundError(f"Vectorizer file not found at {VECT_PATH}")
    return joblib.load(VECT_PATH)

# ===== HELPER FUNCTIONS =====
def plot_bar(counts):
    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {'Positive': '#48bb78', 'Neutral': '#ed8936', 'Negative': '#f56565'}
    colors = [color_map.get(label, '#4299e1') for label in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_title("📊 Sentiment Distribution", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Count", fontweight='bold')
    ax.set_xlabel("Sentiment", fontweight='bold')
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts.values)*0.01 if max(counts.values) > 0 else bar.get_height(), 
                f"{int(v):,}", ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=0)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_pie(counts):
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = {'Positive': '#48bb78', 'Neutral': '#ed8936', 'Negative': '#f56565'}
    colors = [color_map.get(label, '#4299e1') for label in counts.index]
    wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, colors=colors, 
                                     autopct='%1.1f%%', startangle=90, 
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title("🥧 Sentiment Share", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def create_pdf_report(result_df, counts, bar_buf, pie_buf):
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    width, height = A4

    # Title
    title = "Sentiment Analysis Pro - Report"
    c.setFont("Helvetica-Bold", 20)
    c.setFillColorRGB(0.4, 0.2, 0.6)
    c.drawString(50, height - 60, title)

    # Subtitle & time
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(50, height - 80, "AI-powered sentiment analysis report")
    c.setFont("Helvetica", 9)
    c.drawString(50, height - 100, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.3, 0.3, 0.5)
    c.drawString(50, height - 130, "📊 Analysis Summary")
    sentiment_colors = {'Positive': (0.28, 0.73, 0.47), 'Neutral': (0.93, 0.54, 0.21), 'Negative': (0.96, 0.40, 0.40)}
    y_pos = height - 155
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        count = counts.get(sentiment, 0)
        percentage = (count / sum(counts.values)) * 100 if sum(counts.values) > 0 else 0
        c.setFillColorRGB(*sentiment_colors.get(sentiment, (0.4, 0.4, 0.4)))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y_pos, f"{sentiment}:")
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 10)
        c.drawString(140, y_pos, f"{int(count):,} reviews ({percentage:.1f}%)")
        y_pos -= 20

    # Charts
    y_pos -= 30
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.3, 0.3, 0.5)
    c.drawString(50, y_pos, "📈 Visual Analytics")
    try:
        bar_img = ImageReader(bar_buf)
        c.drawImage(bar_img, 50, y_pos - 220, width=250, preserveAspectRatio=True, mask="auto")
    except Exception:
        c.drawString(50, y_pos - 50, "Bar chart not available")
    try:
        pie_img = ImageReader(pie_buf)
        c.drawImage(pie_img, 320, y_pos - 220, width=230, preserveAspectRatio=True, mask="auto")
    except Exception:
        c.drawString(320, y_pos - 50, "Pie chart not available")

    # Data preview
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0.4, 0.2, 0.6)
    c.drawString(50, height - 50, "📋 Data Preview")
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.drawString(50, height - 70, f"First 15 rows of {len(result_df)} total records")
    preview = result_df.head(15)
    cols = preview.columns.tolist()
    cols_to_print = cols[:3] if len(cols) >= 3 else cols
    x_positions = [50 + i * 180 for i in range(len(cols_to_print))]
    y_pos = height - 100
    c.setFillColorRGB(0.8, 0.8, 0.9)
    c.rect(45, y_pos - 5, 530, 15, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0.2, 0.2, 0.4)
    for i, colname in enumerate(cols_to_print):
        c.drawString(x_positions[i], y_pos, str(colname)[:20])
    y_pos -= 20
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0, 0, 0)
    for idx, (_, row) in enumerate(preview.iterrows()):
        if idx % 2 == 0:
            c.setFillColorRGB(0.95, 0.95, 0.95)
            c.rect(45, y_pos - 2, 530, 12, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)
        for i, colname in enumerate(cols_to_print):
            text = str(row[colname])[:25] + "..." if len(str(row[colname])) > 25 else str(row[colname])
            c.drawString(x_positions[i], y_pos, text)
        y_pos -= 12
        if y_pos < 50:
            c.showPage()
            y_pos = height - 50
            c.setFont("Helvetica", 8)
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(50, 30, "Generated by Sentiment Analysis Pro - AI-powered sentiment analysis tool")
    c.save()
    pdf_buf.seek(0)
    return pdf_buf.read()

# ===== LOAD MODEL & VECTORIZER (and handle errors) =====
try:
    model = load_model()
    vectorizer = load_vectorizer()
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: white; font-size: 1.8rem; margin-bottom: 0.5rem;'>😊 Sentiment Analyzer</h1>
        <p style='color: #e2e8f0;'>AI-powered sentiment analysis tool</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.success("✅ Model Loaded")
    st.info(f"*Model Type:* {type(model).__name__}")
    if hasattr(model, 'classes_'):
        st.info(f"*Classes:* {', '.join(model.classes_)}")
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    st.markdown("""
    - *Single Review Analysis*
    - *Batch CSV Processing*
    - *Visual Analytics*
    - *PDF Report Export*
    """)
    st.markdown("---")
    st.markdown("### 🛠 Features")
    st.markdown("""
    - Real-time predictions
    - Multiple chart types
    - Data export options
    - Professional reporting
    """)

# ===== MAIN CONTENT =====
st.markdown('<h1 class="main-header">Sentiment Analysis Pro</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #718096; font-size: 1.2rem; margin-bottom: 3rem;'>
    Analyze sentiment in single reviews or batch process CSV files with beautiful visualizations and professional reports.
</div>
""", unsafe_allow_html=True)

# ===== SINGLE REVIEW PREDICTION =====
st.markdown('<div class="sub-header">🔍 Single Review Analysis</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        single_review = st.text_area(
            "*Enter your review text:*",
            height=140,
            placeholder="Type your review here... We'll analyze the sentiment instantly!",
            help="Enter any product review, comment, or text you want to analyze"
        )
    with col2:
        st.markdown("""
        <div style='background: #f7fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #4299e1;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #2d3748;'>💡 Tips</h4>
            <p style='margin: 0; color: #718096; font-size: 0.9rem;'>
            • Be specific and detailed<br>
            • Include both positive and negative aspects<br>
            • The more text, the better accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

if predict_btn:
    if not single_review.strip():
        st.markdown('<div class="warning-box">⚠ Please enter some text before analyzing sentiment.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("🔍 Analyzing sentiment..."):
            X = vectorizer.transform([single_review])
            pred = model.predict(X)[0]
            sentiment_colors = {'Positive': 'success-box', 'Neutral': 'info-box', 'Negative': 'warning-box'}
            result_class = sentiment_colors.get(pred, 'info-box')
            st.markdown(f'<div class="{result_class}"><h3 style="margin: 0; text-align: center;">Predicted Sentiment: {pred}</h3></div>', unsafe_allow_html=True)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                prob_map = {c: float(p) for c, p in zip(model.classes_, probs)}
                st.markdown("*Confidence Scores:*")
                prob_cols = st.columns(len(prob_map))
                for idx, (sentiment, prob) in enumerate(prob_map.items()):
                    with prob_cols[idx]:
                        st.metric(label=sentiment, value=f"{prob:.1%}", delta="High" if prob > 0.7 else "Medium" if prob > 0.5 else "Low")

st.markdown("---")

# ===== BATCH CSV PROCESSING =====
st.markdown('<div class="sub-header">📁 Batch CSV Analysis</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "*Upload your CSV file:*",
    type=["csv"],
    help="Upload a CSV file containing reviews. We'll auto-detect the review column!"
)

if uploaded is not None:
    # robust CSV read with fallback encodings
    try:
        df = pd.read_csv(uploaded, encoding='utf-8')
    except Exception:
        try:
            df = pd.read_csv(uploaded, encoding='latin-1')
            st.warning("CSV read with latin-1 encoding (non-UTF8 characters were present).")
        except Exception as e:
            st.error(f"❌ Unable to read uploaded CSV: {e}")
            st.stop()

    st.markdown(f'<div class="uploaded-file">✅ File uploaded successfully! <strong>{len(df):,}</strong> rows detected.</div>', unsafe_allow_html=True)

    # ---------- Robust detection & explicit selectbox ----------
    def detect_review_column_robust(df):
        review_keywords = ["review", "text", "comment", "content", "feedback", "opinion", "summary"]
        # 1) exact keyword match
        for col in df.columns:
            lc = col.lower()
            if any(k == lc or lc.endswith(k) or lc.startswith(k) or f"_{k}" in lc for k in review_keywords):
                return col, "keyword-exact"
        # 2) keyword-contained
        for col in df.columns:
            lc = col.lower()
            if any(k in lc for k in review_keywords):
                return col, "keyword-contained"
        # 3) object dtype with largest median length
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if text_cols:
            lengths = {c: df[c].astype(str).map(len).median() for c in text_cols}
            best = max(lengths, key=lengths.get)
            return best, "dtype-object"
        # 4) fallback by median length for all columns
        lengths_all = {c: df[c].astype(str).map(len).median() for c in df.columns}
        best_all = max(lengths_all, key=lengths_all.get)
        return best_all, "fallback-length"

    detected_col, detect_mode = detect_review_column_robust(df)
    st.info(f"Auto-detected review column: **{detected_col}**  (mode: {detect_mode})")

    # candidate columns: object dtype + any with median length > 20
    candidate_cols = []
    for c in df.columns:
        try:
            med_len = df[c].astype(str).map(len).median()
        except Exception:
            med_len = 0
        if df[c].dtype == object or med_len > 20:
            candidate_cols.append(c)

    if detected_col not in candidate_cols:
        candidate_cols.insert(0, detected_col)

    if not candidate_cols:
        candidate_cols = list(df.columns)

    selected_idx = candidate_cols.index(detected_col) if detected_col in candidate_cols else 0
    review_col = st.selectbox("Select review column (auto-detected shown):", options=candidate_cols, index=selected_idx)

    # preview info
    sample_preview = df[review_col].astype(str).dropna().head(5).tolist()
    median_len = int(df[review_col].astype(str).map(len).median() if len(df) > 0 else 0)
    st.markdown(f"**Preview ({review_col})** — median length: **{median_len}** characters")
    st.write(sample_preview)
    if median_len < 10:
        st.warning("Selected column has very short median length (<10). Check if this is the correct review column.")
    if df[review_col].dtype != object:
        st.info(f"Note: Column `{review_col}` is not text dtype — it will be coerced to text before prediction.")

    if st.button("📊 Process Batch Analysis", use_container_width=True):
        with st.spinner("Processing your data... This may take a few moments"):
            X = vectorizer.transform(df[review_col].astype(str))
            df["Predicted_Sentiment"] = model.predict(X)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                df["Confidence_Score"] = [float(p.max()) for p in probs]
            counts = df["Predicted_Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            bar_buf = plot_bar(counts)
            pie_buf = plot_pie(counts)

        st.success(f"✅ Analysis complete! Processed {len(df)} reviews.")

        # Display metrics
        st.markdown("### 📈 Analysis Results")
        metric_cols = st.columns(3)
        total = len(df) if len(df) > 0 else 1

        with metric_cols[0]:
            positive_pct = (counts.get("Positive", 0) / total) * 100
            st.metric("😊 Positive", f"{counts.get('Positive', 0):,}", f"{positive_pct:.1f}%")

        with metric_cols[1]:
            neutral_pct = (counts.get("Neutral", 0) / total) * 100
            st.metric("😐 Neutral", f"{counts.get('Neutral', 0):,}", f"{neutral_pct:.1f}%")

        with metric_cols[2]:
            negative_pct = (counts.get("Negative", 0) / total) * 100
            st.metric("😞 Negative", f"{counts.get('Negative', 0):,}", f"{negative_pct:.1f}%")

        # Show charts
        st.markdown("### 📊 Visualizations")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.image(bar_buf, use_column_width=True, caption="Sentiment Distribution")
        with viz_col2:
            st.image(pie_buf, use_column_width=True, caption="Sentiment Share")

        # Results table
        st.markdown("### 📋 Results Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Export section
        st.markdown("### 💾 Export Results")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV",
                data=csv_bytes,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with export_col2:
            with st.spinner("Generating PDF report..."):
                pdf_bytes = create_pdf_report(df, counts, bar_buf, pie_buf)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 2rem;'>
    <p>Built By Dasaradh | Sentiment Analysis Pro v1.0</p>
</div>
""", unsafe_allow_html=True)
