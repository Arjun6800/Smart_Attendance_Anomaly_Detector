"""
app.py
=======
Smart Attendance Anomaly Detector – Interactive Streamlit Web Application

Pages
-----
1. Dashboard        – Summary statistics and top suspicious students
2. Upload & Detect  – Upload CSV, run all three models, view flagged records
3. Student Analysis – Deep-dive into a single student's attendance behaviour
4. Algorithm Comparison – Metrics table and performance charts
5. Visualisations   – Browse all generated plots
6. Model Information – Algorithm explanations and project documentation

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root on sys.path ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance Anomaly Detector",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252836);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #7c4dff;
        margin-bottom: 1rem;
    }
    .anomaly-badge {
        background-color: #f44336;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .normal-badge {
        background-color: #4caf50;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    h1, h2, h3 { color: #e0e0e0; }
    .stDataFrame { border-radius: 8px; }
    .sidebar .sidebar-content { background-color: #1a1d2e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR   = os.path.join(ROOT, "results", "plots")
MODELS_DIR  = os.path.join(ROOT, "models")
DATA_DIR    = os.path.join(ROOT, "data")

# ─────────────────────────────────────────────
# Helper: Load saved artefacts
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load pre-trained models from disk. Returns None if not trained yet."""
    paths = {
        "if":     os.path.join(MODELS_DIR, "isolation_forest.pkl"),
        "lof":    os.path.join(MODELS_DIR, "lof_model.pkl"),
        "hybrid": os.path.join(MODELS_DIR, "hybrid_model.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
        "stats":  os.path.join(MODELS_DIR, "train_stats.pkl"),
    }
    loaded = {}
    for name, path in paths.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded


@st.cache_data(show_spinner=False)
def load_results():
    """Load CSV result files."""
    results = {}
    csv_files = {
        "comparison":    "model_comparison.csv",
        "hyperparams":   "hyperparameter_results.csv",
        "threshold":     "threshold_results.csv",
        "alpha":         "alpha_results.csv",
        "predictions":   "test_predictions.csv",
    }
    for key, fname in csv_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            try:
                results[key] = pd.read_csv(path, index_col=0 if key == "comparison" else None)
            except Exception:
                results[key] = None
    return results


@st.cache_data(show_spinner=False)
def load_demo_data():
    demo_path = os.path.join(DATA_DIR, "demo_attendance.csv")
    if os.path.exists(demo_path):
        return pd.read_csv(demo_path)
    return None


# ─────────────────────────────────────────────
# Helper: Run prediction on uploaded data
# ─────────────────────────────────────────────

def predict_on_dataframe(df_input: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Run all three anomaly detectors on a new attendance DataFrame.
    Returns the DataFrame with added prediction and score columns.
    """
    from src.preprocessing       import clean_data
    from src.feature_engineering import engineer_features, scale_features

    required = [
        "student_id", "date", "day_of_week", "class_id",
        "attendance_status", "attendance_time", "hour", "minute",
        "location_id", "ip_address", "device_id",
        "previous_attendance_gap", "weekly_attendance_rate",
        "monthly_attendance_rate", "total_classes", "classes_attended",
        "late_count", "absence_count", "proxy_risk_indicator",
        "historical_deviation",
    ]

    # Add is_anomaly placeholder if missing (for compatibility)
    if "is_anomaly" not in df_input.columns:
        df_input = df_input.copy()
        df_input["is_anomaly"]  = 0
        df_input["anomaly_type"] = "unknown"

    missing = [c for c in required if c not in df_input.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None

    try:
        df_clean = clean_data(df_input)
        train_stats = models.get("stats")
        if train_stats is None:
            st.warning("Training statistics not found. Re-computing from uploaded data.")
            X_fe, stats = engineer_features(df_clean, fit=True)
        else:
            X_fe, _ = engineer_features(
                df_clean,
                student_stats=train_stats["student_stats"],
                sharing_stats=train_stats["sharing_stats"],
                fit=False,
            )

        scaler = models.get("scaler")
        if scaler is not None:
            X_scaled = scaler.transform(X_fe)
        else:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_fe)

        # ── Run detectors ─────────────────────────────────────────────
        result_df = df_clean.copy()

        if "if" in models:
            from src.models.isolation_forest_model import IsolationForestDetector
            if_det = IsolationForestDetector()
            if_det._model = models["if"]
            result_df["if_pred"],  result_df["if_score"]  = if_det.predict_with_scores(X_scaled)

        if "lof" in models:
            from src.models.lof_model import LOFDetector
            lof_det = LOFDetector()
            lof_det._model = models["lof"]
            result_df["lof_pred"], result_df["lof_score"] = lof_det.predict_with_scores(X_scaled)

        if "hybrid" in models and "if" in models and "lof" in models:
            hybrid = models["hybrid"]
            result_df["hybrid_pred"], result_df["hybrid_score"] = hybrid.predict_with_scores(
                result_df["if_score"].values,
                result_df["lof_score"].values,
            )

        # ── Add risk level and explanation ────────────────────────────
        from src.evaluate import explain_anomaly

        def risk_level(score):
            if score >= 0.80:  return "🔴 HIGH"
            if score >= 0.60:  return "🟠 MEDIUM"
            if score >= 0.40:  return "🟡 LOW"
            return "🟢 NORMAL"

        if "hybrid_score" in result_df.columns:
            result_df["risk_level"]  = result_df["hybrid_score"].apply(risk_level)
            result_df["explanation"] = result_df.apply(explain_anomaly, axis=1)

        return result_df

    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────

st.sidebar.image(
    "https://img.icons8.com/color/96/000000/graduation-cap.png",
    width=80
)
st.sidebar.title("🎓 Attendance\nAnomaly Detector")
st.sidebar.markdown("---")

PAGE = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🔍 Upload & Detect",
        "👤 Student Analysis",
        "📈 Algorithm Comparison",
        "🖼️ Visualisations",
        "ℹ️ Model Information",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Smart Attendance Anomaly Detector**")
st.sidebar.markdown("Academic AI Project | 30 Marks")
st.sidebar.markdown("Algorithms: IF · LOF · Hybrid")


# ─────────────────────────────────────────────
# Load models and results
# ─────────────────────────────────────────────

models  = load_models()
results = load_results()
models_loaded = bool(models)

if not models_loaded:
    st.warning(
        "⚠️ Pre-trained models not found. "
        "Run `python run_project.py` first to train models and generate results."
    )


# ─────────────────────────────────────────────
# Helper display function (must be defined before page code)
# ─────────────────────────────────────────────

def _show_results(result: pd.DataFrame):
    """Helper to display prediction results."""
    n_flagged = int(result["hybrid_pred"].sum()) \
                if "hybrid_pred" in result.columns else 0
    st.success(f"Detection complete! {n_flagged} anomalies found in {len(result)} records.")
    display_cols = [c for c in [
        "student_id", "date", "hour", "weekly_attendance_rate",
        "if_score", "lof_score", "hybrid_score", "risk_level", "explanation"
    ] if c in result.columns]
    flagged = result[result["hybrid_pred"] == 1] if "hybrid_pred" in result.columns else result
    st.dataframe(flagged[display_cols].head(50), use_container_width=True)
    csv = result.to_csv(index=False).encode()
    st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")


# ═════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═════════════════════════════════════════════

if PAGE == "📊 Dashboard":
    st.title("📊 Smart Attendance Anomaly Detector")
    st.markdown("**AI-powered detection of suspicious attendance patterns using Isolation Forest, LOF, and a Student-Designed Hybrid Model.**")
    st.markdown("---")

    # ── Summary metrics from test predictions ────────────────────────────
    pred_df = results.get("predictions")
    comp_df = results.get("comparison")

    if pred_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        total = len(pred_df)
        n_anomaly = int(pred_df.get("hybrid_pred", pd.Series([0])).sum()) \
                    if "hybrid_pred" in pred_df.columns else 0
        n_students = pred_df["student_id"].nunique() if "student_id" in pred_df.columns else "–"
        avg_rate = pred_df["weekly_attendance_rate"].mean() if "weekly_attendance_rate" in pred_df.columns else 0

        col1.metric("📋 Total Records (Test)", f"{total:,}")
        col2.metric("👥 Students", str(n_students))
        col3.metric("🚨 Detected Anomalies", f"{n_anomaly:,}")
        col4.metric("📉 Anomaly Rate", f"{n_anomaly/total*100:.1f}%")

        st.markdown("---")

        # Top suspicious students
        if "student_id" in pred_df.columns and "hybrid_score" in pred_df.columns:
            st.subheader("🏴 Most Suspicious Students")
            top = (
                pred_df.groupby("student_id")["hybrid_score"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            top.columns = ["Student ID", "Avg Hybrid Score"]
            top["Risk Level"] = top["Avg Hybrid Score"].apply(
                lambda s: "🔴 HIGH" if s >= 0.8 else ("🟠 MEDIUM" if s >= 0.6 else "🟡 LOW")
            )
            st.dataframe(top, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Recent Flagged Records")
        if "hybrid_pred" in pred_df.columns:
            flagged = pred_df[pred_df["hybrid_pred"] == 1].head(20)
            display_cols = [c for c in [
                "student_id", "date", "hour", "weekly_attendance_rate",
                "hybrid_score", "risk_level", "explanation"
            ] if c in flagged.columns]
            st.dataframe(flagged[display_cols], use_container_width=True)
    else:
        st.info("No results found. Run `python run_project.py` to generate results.")

    if comp_df is not None:
        st.markdown("---")
        st.subheader("📈 Algorithm Performance Summary")
        st.dataframe(comp_df, use_container_width=True)

    # Show plot previews
    if os.path.isdir(PLOTS_DIR):
        plots = [f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
        if plots:
            st.markdown("---")
            st.subheader("🖼️ Plot Gallery Preview")
            cols = st.columns(3)
            for i, plot in enumerate(plots[:6]):
                with cols[i % 3]:
                    st.image(os.path.join(PLOTS_DIR, plot),
                             caption=plot.replace(".png", "").replace("_", " ").title(),
                             use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 2: UPLOAD & DETECT
# ═════════════════════════════════════════════

elif PAGE == "🔍 Upload & Detect":
    st.title("🔍 Upload Attendance Data & Detect Anomalies")
    st.markdown("Upload a CSV file or use the built-in demo data to run anomaly detection.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📂 Upload CSV", "🎯 Demo Data"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload Attendance CSV",
            type=["csv"],
            help="CSV must contain the same columns as the training data."
        )
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                st.success(f"✓ Loaded {len(df_up):,} records from uploaded file.")
                st.dataframe(df_up.head(5), use_container_width=True)

                if st.button("🚀 Run Anomaly Detection", type="primary"):
                    if not models_loaded:
                        st.error("Models not loaded. Run `python run_project.py` first.")
                    else:
                        with st.spinner("Running anomaly detection …"):
                            result = predict_on_dataframe(df_up, models)
                        if result is not None:
                            _show_results(result)
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with tab2:
        demo_df = load_demo_data()
        if demo_df is not None:
            st.info(f"Demo dataset: {len(demo_df):,} records loaded.")
            st.dataframe(demo_df.head(10), use_container_width=True)

            # Filter controls
            st.markdown("**Filter Options**")
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                if "student_id" in demo_df.columns:
                    students = ["All"] + sorted(demo_df["student_id"].unique().tolist())
                    sel_student = st.selectbox("Filter by Student", students)
            with filter_col2:
                risk_filter = st.selectbox("Risk Level Filter (after detection)",
                                           ["All", "🔴 HIGH", "🟠 MEDIUM", "🟡 LOW", "🟢 NORMAL"])

            if st.button("🚀 Run Anomaly Detection on Demo Data", type="primary"):
                if not models_loaded:
                    st.error("Models not loaded. Run `python run_project.py` first.")
                else:
                    df_to_run = demo_df.copy()
                    if sel_student != "All":
                        df_to_run = df_to_run[df_to_run["student_id"] == sel_student]

                    with st.spinner("Running anomaly detection …"):
                        result = predict_on_dataframe(df_to_run, models)

                    if result is not None:
                        # Apply risk filter
                        if risk_filter != "All" and "risk_level" in result.columns:
                            result = result[result["risk_level"] == risk_filter]

                        n_flagged = int(result.get("hybrid_pred", pd.Series([0])).sum()) \
                                    if "hybrid_pred" in result.columns else 0

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Records Processed", len(result))
                        col2.metric("Anomalies Flagged", n_flagged)
                        col3.metric("Anomaly Rate", f"{n_flagged/max(len(result),1)*100:.1f}%")

                        st.markdown("---")
                        st.subheader("🚨 Detected Anomalies")

                        if "hybrid_pred" in result.columns:
                            flagged = result[result["hybrid_pred"] == 1].sort_values(
                                "hybrid_score", ascending=False
                            ) if "hybrid_score" in result.columns else result[result["hybrid_pred"] == 1]
                        else:
                            flagged = result

                        display_cols = [c for c in [
                            "student_id", "date", "hour", "weekly_attendance_rate",
                            "if_score", "lof_score", "hybrid_score",
                            "risk_level", "explanation"
                        ] if c in flagged.columns]
                        st.dataframe(flagged[display_cols], use_container_width=True)

                        # Download button
                        csv_bytes = result.to_csv(index=False).encode()
                        st.download_button(
                            "📥 Download Full Results CSV",
                            csv_bytes,
                            file_name="anomaly_predictions.csv",
                            mime="text/csv",
                        )
        else:
            st.warning("Demo data not found. Run `python data/generate_dataset.py` first.")




# ═════════════════════════════════════════════
# PAGE 3: STUDENT ANALYSIS
# ═════════════════════════════════════════════

elif PAGE == "👤 Student Analysis":
    st.title("👤 Student Attendance Analysis")
    st.markdown("---")

    pred_df = results.get("predictions")
    demo_df = load_demo_data()

    # Prefer predictions if available, else demo
    df_base = pred_df if pred_df is not None else demo_df

    if df_base is not None and "student_id" in df_base.columns:
        students = sorted(df_base["student_id"].unique().tolist())
        selected = st.selectbox("Select Student ID", students)

        stu_df = df_base[df_base["student_id"] == selected].copy()

        col1, col2, col3, col4 = st.columns(4)
        avg_rate = stu_df["weekly_attendance_rate"].mean() if "weekly_attendance_rate" in stu_df else 0
        n_records = len(stu_df)
        n_anom = int(stu_df["hybrid_pred"].sum()) if "hybrid_pred" in stu_df.columns else "–"
        avg_score = stu_df["hybrid_score"].mean() if "hybrid_score" in stu_df.columns else 0

        col1.metric("Records", n_records)
        col2.metric("Avg Attendance Rate", f"{avg_rate:.1%}")
        col3.metric("Anomalies Flagged", str(n_anom))
        col4.metric("Avg Hybrid Score", f"{avg_score:.4f}")

        st.markdown("---")

        tab1, tab2 = st.tabs(["📋 Record Table", "📈 Attendance Timeline"])
        with tab1:
            show_cols = [c for c in [
                "date", "hour", "weekly_attendance_rate", "historical_deviation",
                "if_score", "lof_score", "hybrid_score", "risk_level", "explanation"
            ] if c in stu_df.columns]
            st.dataframe(stu_df[show_cols].sort_values("date") if "date" in stu_df.columns
                         else stu_df[show_cols], use_container_width=True)

        with tab2:
            if "date" in stu_df.columns and "weekly_attendance_rate" in stu_df.columns:
                stu_df["date"] = pd.to_datetime(stu_df["date"], errors="coerce")
                stu_df = stu_df.dropna(subset=["date"]).sort_values("date")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(stu_df["date"], stu_df["weekly_attendance_rate"],
                        color="steelblue", lw=1.5, alpha=0.8)

                if "hybrid_pred" in stu_df.columns:
                    anomaly_rows = stu_df[stu_df["hybrid_pred"] == 1]
                    normal_rows  = stu_df[stu_df["hybrid_pred"] == 0]
                    ax.scatter(normal_rows["date"],  normal_rows["weekly_attendance_rate"],
                               color="#4caf50", s=40, zorder=5, label="Normal")
                    ax.scatter(anomaly_rows["date"], anomaly_rows["weekly_attendance_rate"],
                               color="#f44336", s=100, marker="X", zorder=6, label="Anomaly")

                ax.set_title(f"Attendance Rate Timeline – {selected}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Weekly Attendance Rate")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ── Suspicious events summary ─────────────────────────────────────
        if "hybrid_pred" in stu_df.columns:
            suspicious = stu_df[stu_df["hybrid_pred"] == 1]
            if len(suspicious) > 0:
                st.markdown("---")
                st.subheader("⚠️ Suspicious Events")
                reason_col = [c for c in ["explanation", "risk_level"] if c in suspicious.columns]
                st.dataframe(suspicious[reason_col + ["date", "hour"]].head(10) if "date" in suspicious.columns
                             else suspicious[reason_col].head(10),
                             use_container_width=True)
    else:
        st.info("Run `python run_project.py` to generate test predictions, or use Demo data.")


# ═════════════════════════════════════════════
# PAGE 4: ALGORITHM COMPARISON
# ═════════════════════════════════════════════

elif PAGE == "📈 Algorithm Comparison":
    st.title("📈 Algorithm Performance Comparison")
    st.markdown("---")

    comp_df = results.get("comparison")
    if comp_df is not None:
        st.subheader("📊 Test Set Metrics")
        st.dataframe(comp_df, use_container_width=True)

        # Bar chart using matplotlib
        st.markdown("---")
        st.subheader("📊 Visual Comparison")

        metrics = [c for c in ["precision", "recall", "f1", "roc_auc", "pr_auc"]
                   if c in comp_df.columns]
        models_list = comp_df.index.tolist()
        x   = np.arange(len(metrics))
        w   = 0.25
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0e1117")
        ax.set_facecolor("#1a1d2e")
        colors = ["#2196F3", "#FF9800", "#9C27B0"]

        for i, (model, color) in enumerate(zip(models_list, colors)):
            vals = [float(comp_df.loc[model, m]) for m in metrics]
            offset = (i - len(models_list) / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=model, color=color, alpha=0.85,
                          edgecolor="#0e1117")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom",
                        color="white", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper().replace("_", "-") for m in metrics], color="white")
        ax.set_ylim([0, 1.15])
        ax.set_ylabel("Score", color="white")
        ax.set_title("Algorithm Performance Comparison (Test Set)", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1d2e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Run `python run_project.py` to generate comparison results.")

    # Alpha experiment
    alpha_df = results.get("alpha")
    if alpha_df is not None:
        st.markdown("---")
        st.subheader("🔬 Hybrid Model Alpha Tuning (Student Innovation)")
        st.dataframe(alpha_df, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
        ax2.set_facecolor("#1a1d2e")
        ax2.plot(alpha_df["alpha"], alpha_df["f1"],
                 marker="o", color="#9C27B0", lw=2, markersize=8)
        best = alpha_df.loc[alpha_df["f1"].idxmax()]
        ax2.axvline(x=best["alpha"], color="red", ls="--", lw=1.5,
                    label=f"Best α={best['alpha']:.2f}  F1={best['f1']:.4f}")
        ax2.set_xlabel("Alpha", color="white")
        ax2.set_ylabel("F1-Score", color="white")
        ax2.set_title("Hybrid Alpha Experiment (Validation F1)", color="white")
        ax2.tick_params(colors="white")
        ax2.legend(facecolor="#1a1d2e", labelcolor="white")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Hyperparameter results
    hp_df = results.get("hyperparams")
    if hp_df is not None:
        st.markdown("---")
        st.subheader("🔬 Hyperparameter Search Results")
        st.dataframe(hp_df, use_container_width=True)

    # Threshold results
    thr_df = results.get("threshold")
    if thr_df is not None:
        st.markdown("---")
        st.subheader("🎚️ Threshold Sweep (Isolation Forest on Validation Set)")
        st.dataframe(thr_df, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 5: VISUALISATIONS
# ═════════════════════════════════════════════

elif PAGE == "🖼️ Visualisations":
    st.title("🖼️ Generated Visualisations")
    st.markdown("All plots generated during model training and evaluation.")
    st.markdown("---")

    if os.path.isdir(PLOTS_DIR):
        plots = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")])
        if plots:
            for i in range(0, len(plots), 2):
                col1, col2 = st.columns(2)
                for j, col in enumerate([col1, col2]):
                    if i + j < len(plots):
                        fname = plots[i + j]
                        with col:
                            st.image(
                                os.path.join(PLOTS_DIR, fname),
                                caption=fname.replace(".png", "").replace("_", " ").title(),
                                use_container_width=True,
                            )
        else:
            st.info("No plots found. Run `python run_project.py` to generate plots.")
    else:
        st.info("Plots directory not found. Run `python run_project.py` first.")


# ═════════════════════════════════════════════
# PAGE 6: MODEL INFORMATION
# ═════════════════════════════════════════════

elif PAGE == "ℹ️ Model Information":
    st.title("ℹ️ Model Information & Documentation")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌲 Isolation Forest",
        "🔵 Local Outlier Factor",
        "🔮 Hybrid Model",
        "📄 Ethics & Privacy",
    ])

    with tab1:
        st.subheader("🌲 Isolation Forest")
        st.markdown("""
**Algorithm Type:** Tree-based global anomaly detection  
**Paper:** Liu et al., 2008 – "Isolation Forest"

### How it Works
Isolation Forest builds an ensemble of random **Isolation Trees**.  
Each tree randomly selects a feature and a split value to partition the data.

**Key Insight:** Anomalous points are **rare and different**, so they get
isolated (separated from all other points) in very **few splits** (short path length).

Normal points require many splits because they cluster together.

### Score Interpretation
- Score closer to **1** → short isolation path → **anomaly**
- Score around **0.5** → borderline / uncertain
- Score much less than **0.5** → normal

### Parameters
| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of isolation trees (more = stable, slower) |
| `contamination` | Expected fraction of anomalies (guides threshold) |
| `max_samples` | Samples per tree (`auto` = min(256, n)) |

### Strengths
- Works well on high-dimensional data
- Computationally efficient: O(n log n)
- Naturally handles global outliers

### Weaknesses
- Less effective at detecting local anomalies (clusters with local density)
- Performance can degrade if contamination is set incorrectly
""")

    with tab2:
        st.subheader("🔵 Local Outlier Factor (LOF)")
        st.markdown("""
**Algorithm Type:** Density-based local anomaly detection  
**Paper:** Breunig et al., 2000 – "LOF: Identifying Density-Based Local Outliers"

### How it Works
LOF compares the **local density** of each point to the local density of its **k-nearest neighbours**.

1. **k-NN:** Find k nearest neighbours for each point
2. **Local Reachability Density (LRD):** Inverse of average reachability distance
3. **LOF score:** Ratio of neighbours' LRD to point's LRD

### Score Interpretation
- LOF ≈ **1** → similar density to neighbours → **normal**
- LOF >> **1** → much sparser than neighbours → **anomaly**

### Parameters
| Parameter | Description |
|-----------|-------------|
| `n_neighbors` | k for kNN (larger k = smoother density) |
| `contamination` | Expected anomaly fraction |
| `novelty=True` | Required for predicting on new data |

### Strengths
- Excellent at detecting **local** anomalies
- Works in irregular cluster shapes
- Provides interpretable density scores

### Weaknesses
- O(n²) complexity for exact kNN
- Performance degrades in very high dimensions
- Requires careful selection of k (n_neighbors)
""")

    with tab3:
        st.subheader("🔮 Hybrid Model – Student-Designed Innovation")
        st.markdown("""
### Motivation
Isolation Forest and LOF have **complementary strengths**:
- IF is better at global anomalies
- LOF is better at local anomalies

The hybrid model attempts to **combine both** for improved overall detection.

### Formula
```
HybridScore = α × IF_score_norm + (1 - α) × LOF_score_norm
```

Where:
- `IF_score_norm` = IF score normalised to [0, 1]
- `LOF_score_norm` = LOF score normalised to [0, 1]
- `α` = weighting parameter (tuned on validation set)

### Score Normalisation
Both scores are normalised using training-set min/max to ensure fair combination:
```
score_norm = (score - min_train) / (max_train - min_train)
```
Training statistics are used to avoid data leakage.

### Alpha Tuning
Alpha is selected by maximising **F1-score on the validation set**.  
Candidate values: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

### Threshold Tuning
Anomaly decision threshold is also tuned on the validation set.  
Candidate values: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

### Limitations
- Improvement is not guaranteed – depends on dataset characteristics
- If IF and LOF errors are correlated, the hybrid may not help
- Adding a bad model can degrade a good one
""")

    with tab4:
        st.subheader("📄 Ethics & Privacy")
        st.markdown("""
### ⚠️ Important Ethical Considerations

#### Synthetic Data
This project uses **entirely synthetic data**. No real student attendance
records were used or exposed.

#### Anomaly ≠ Fraud
An anomaly flag from this system **does NOT prove misconduct**.  
It indicates an unusual pattern that warrants **human review**.

A student could be flagged due to:
- Technical errors in the attendance system
- Legitimate schedule changes
- System clock differences
- Network or device issues

#### Human Review Required
Before taking any disciplinary action:
1. A qualified human reviewer must examine the flagged records
2. The student must be given an opportunity to explain
3. Multiple sources of evidence should be considered

#### False Positives
All anomaly detection systems produce false positives.
The threshold setting controls the precision-recall trade-off.

#### Data Protection
- Real attendance data is personally identifiable information (PII)
- It must be stored securely and accessed only by authorised personnel
- Retention policies must comply with institutional and legal requirements

#### Bias Considerations
The system may be biased if:
- The training data is not representative
- Legitimate behavioural differences exist across student groups
- The anomaly features contain proxies for protected characteristics
""")
