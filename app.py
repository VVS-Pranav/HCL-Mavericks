"""NextBill dashboard. Start with: python -m streamlit run app.py"""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="NextBill | Customer forecasts", page_icon="📊", layout="wide")
st.markdown("""<style>
    .stApp { background: #f7f9fb; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e1e7ed; padding: 18px; border-radius: 12px; }
    h1, h2, h3 { color: #163443; }
    .block-container { padding-top: 2rem; }
</style>""", unsafe_allow_html=True)


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_tables():
    """Read current notebook outputs again on every interaction."""
    monitoring = ROOT / "monitoring_outputs"
    models = ROOT / "model_outputs"
    inputs = ROOT / "dashboard_inputs"
    return {
        "customers": pd.read_csv(monitoring / "latest_customer_monitoring.csv"),
        "drivers": pd.read_csv(monitoring / "latest_group_contributions.csv"),
        "history": pd.read_csv(inputs / "customer_history.csv", parse_dates=["billing_month"]),
        "features": pd.read_csv(inputs / "latest_customer_features.csv"),
        "model_summary": read_json(models / "model_summary.json"),
        "monitoring_summary": read_json(monitoring / "monitoring_summary.json"),
        "test_metrics": pd.read_csv(models / "test_metrics.csv"),
        "validation": pd.read_csv(models / "validation_model_comparison.csv"),
        "segments": pd.read_csv(models / "test_segment_metrics.csv"),
        "importance": pd.read_csv(models / "validation_feature_importance.csv"),
        "alerts": pd.read_csv(monitoring / "alert_quality.csv"),
        "intervals": pd.read_csv(monitoring / "interval_metrics_by_group.csv"),
    }


try:
    tables = load_tables()
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error("Required dashboard files are missing or invalid. Keep the supplied folders beside app.py, or run Steps 1–3 and python sync_dashboard_inputs.py.")
    st.code(str(error))
    st.stop()

customers = tables["customers"]
model_summary = tables["model_summary"]
monitoring_summary = tables["monitoring_summary"]
forecast_month = pd.to_datetime(customers["forecast_month"].max()).strftime("%B %Y")

st.sidebar.title("NextBill")
st.sidebar.caption("HCL hackathon · Telecom analytics")
page = st.sidebar.radio("Navigate", ["Customer forecasts", "Model performance", "Method and setup"])
st.sidebar.divider()
st.sidebar.write("**One customer. One next-month forecast.**")
st.sidebar.caption("Forecasts use completed monthly bills. Amounts are in dataset bill units; currency is unspecified.")
st.sidebar.caption("Synthetic dataset · Existing customers")

if page == "Customer forecasts":
    st.title("Customer forecasts")
    st.caption(f"{forecast_month} · Individual forecasts from {model_summary['recommended_forecast']} · Latest outcomes are unverified")
    cards = st.columns(4)
    cards[0].metric("Customers", f"{len(customers):,}")
    cards[1].metric("Total predicted bills", f"{customers['predicted_next_bill'].sum():,.2f}")
    cards[2].metric("Point high-bill alerts", int(customers["high_bill_alert"].sum()))
    cards[3].metric("Historical range coverage", f"{monitoring_summary['interval_evaluation']['coverage_percent']:.2f}%")
    st.caption("Coverage is measured on April–May 2025 against a 90% target; it is not a confidence score for an individual customer.")

    search_col, plan_col, priority_col = st.columns([2, 1, 1.4])
    search = search_col.text_input("Search customer ID", placeholder="For example, C00001", key="customer_search")
    plan = plan_col.selectbox("Plan", ["All plans"] + sorted(customers["plan_type"].unique()), key="plan_filter")
    priorities = ["All priorities", "Strong alert", "High-bill alert", "History review", "Possible high bill", "Within range"]
    priority = priority_col.selectbox("Review priority", priorities, key="priority_filter")
    filtered = customers.copy()
    if search.strip():
        filtered = filtered.loc[filtered["customer_id"].str.contains(search.strip(), case=False, regex=False)]
    if plan != "All plans":
        filtered = filtered.loc[filtered["plan_type"].eq(plan)]
    if priority != "All priorities":
        filtered = filtered.loc[filtered["priority"].eq(priority)]
    st.subheader(f"Review queue · {len(filtered):,} customers")
    columns = ["customer_id", "plan_type", "predicted_next_bill", "predicted_increase_percent", "priority"]
    st.dataframe(filtered[columns].round(2), hide_index=True, width="stretch")
    st.download_button("Download filtered customers", filtered.to_csv(index=False).encode(),
                       file_name="nextbill_filtered_customers.csv", mime="text/csv")
    if filtered.empty:
        st.info("No customers match these filters. Clear the search or choose another plan or priority.")
        st.stop()

    selected_id = st.selectbox("Select a customer to inspect", filtered["customer_id"].tolist(), key="selected_customer")
    row = filtered.loc[filtered["customer_id"].eq(selected_id)].iloc[0]
    st.subheader(f"{selected_id} · {row['plan_type']} plan")
    cards = st.columns(3)
    cards[0].metric("Predicted next bill", f"{row['predicted_next_bill']:,.2f}", f"{row['predicted_increase_percent']:+.1f}% vs recent average", delta_color="inverse")
    cards[1].metric("Estimated bill range", f"{row['estimated_lower_bill']:,.2f} – {row['estimated_upper_bill']:,.2f}")
    cards[2].metric("High-bill threshold", f"{row['high_bill_threshold']:,.2f}")
    st.info(f"{row['priority']}: {row['suggested_review']}")

    history = tables["history"].loc[tables["history"]["customer_id"].eq(selected_id)].sort_values("billing_month")
    st.write("**Completed bill history and next-month forecast**")
    next_month = pd.Timestamp(row["forecast_month"])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(history["billing_month"], history["bill_amount"], color="#247b76", marker=".", label="Completed bill")
    ax.plot([history["billing_month"].iloc[-1], next_month],
            [history["bill_amount"].iloc[-1], row["predicted_next_bill"]], color="#d67b29", linestyle="--")
    ax.errorbar(next_month, row["predicted_next_bill"],
                yerr=[[row["predicted_next_bill"] - row["estimated_lower_bill"]],
                      [row["estimated_upper_bill"] - row["predicted_next_bill"]]],
                fmt="o", capsize=6, color="#d67b29", label="Forecast and estimated range")
    ax.set_ylabel("Bill units")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Recent three-bill average: {row['recent_average']:,.2f}. Recorded history: {int(row['history_months'])} months.")

    st.write("**What influenced this prediction?**")
    st.caption("Ordered feature-group comparisons with a typical pre-test record on the same plan. The reference is separate from this customer's recent average. Results depend on group order and do not prove billing causes.")
    drivers = tables["drivers"].loc[tables["drivers"]["customer_id"].eq(selected_id)].iloc[0]
    groups = [column for column in tables["drivers"].columns if column not in ["customer_id", "forecast_month", "reference_prediction"]]
    contributions = drivers[groups].astype(float).sort_values()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.barh(contributions.index, contributions.values,
            color=["#d67b29" if value > 0 else "#247b76" for value in contributions])
    ax.axvline(0, color="#7b8991", linewidth=0.7)
    ax.set_xlabel("Change in model prediction (bill units)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Reference prediction {drivers['reference_prediction']:,.2f} + contributions {contributions.sum():+.2f} = {row['predicted_next_bill']:,.2f}.")

    if st.button("Run saved model for this customer", key="run_model"):
        feature_rows = tables["features"].loc[tables["features"]["customer_id"].eq(selected_id)]
        if len(feature_rows) != 1:
            st.error("Expected one latest feature record. Run python sync_dashboard_inputs.py after rerunning your notebooks.")
        else:
            try:
                bundle = joblib.load(ROOT / "model_outputs/forecast_bundle.joblib")
                inputs = feature_rows[bundle["feature_columns"]]
                if bundle["use_baseline"]:
                    prediction = float(inputs[bundle["baseline_column"]].iloc[0])
                else:
                    with threadpool_limits(limits=2):
                        prediction = float(bundle["pipeline"].predict(inputs)[0])
                prediction = max(0, prediction)
                st.success(f"Saved model prediction for {selected_id}: {prediction:,.2f} bill units.")
                if not np.isclose(prediction, row["predicted_next_bill"], atol=0.01):
                    st.warning("The model and displayed outputs differ. Rerun Step 3 and sync the dashboard inputs.")
            except (FileNotFoundError, ValueError, KeyError, ImportError, AttributeError) as error:
                st.error(f"Could not run the saved model: {error}")
    st.download_button("Download this customer", filtered.loc[filtered["customer_id"].eq(selected_id)].to_csv(index=False).encode(),
                       file_name=f"{selected_id}_forecast.csv", mime="text/csv")

elif page == "Model performance":
    st.title("Model performance")
    st.caption("Point accuracy: March–May 2025 · Interval and alert checks: April–May 2025")
    scores = model_summary["test_metrics_for_recommendation"]
    cards = st.columns(4)
    cards[0].metric("Test MAE", f"{scores['MAE']:.2f}")
    cards[1].metric("Test RMSE", f"{scores['RMSE']:.2f}")
    cards[2].metric("Test WAPE", f"{scores['WAPE_percent']:.2f}%")
    cards[3].metric("MAE improvement vs baseline", f"{model_summary['test_improvement_percent']:.2f}%")
    st.caption("MAE and RMSE are in bill units. WAPE measures error; it is not an accuracy percentage.")
    st.subheader("Chronological test comparison")
    st.dataframe(tables["test_metrics"].round(3), hide_index=True, width="stretch")
    with st.expander("Validation model selection"):
        st.dataframe(tables["validation"].round(3), hide_index=True, width="stretch")
    st.subheader("Segmented test errors")
    segment = st.selectbox("Group errors by", tables["segments"]["segment"].unique().tolist(), key="segment_filter")
    st.dataframe(tables["segments"].loc[tables["segments"]["segment"].eq(segment)].round(3), hide_index=True, width="stretch")
    st.subheader("Estimated range reliability")
    st.dataframe(tables["intervals"].round(3), hide_index=True, width="stretch")
    st.caption("March residuals calibrate the range. The later measured coverage is 88.35%, compared with the 90% target. Repeated customers and drift limit formal coverage claims.")
    st.subheader("High-bill detection and review workload")
    st.dataframe(tables["alerts"].round(3), hide_index=True, width="stretch")
    point, watch = monitoring_summary["alert_evaluation"]
    st.warning(f"Point alerts caught {point['true_positive']} of {point['actual_high_rows']} actual high bills. The upper-range watchlist caught {watch['true_positive']}, while flagging {watch['review_workload_percent']:.1f}% of records. Sudden-spike detection remains limited.")
    st.subheader("Portfolio feature importance")
    st.dataframe(tables["importance"].head(10).round(3), hide_index=True, width="stretch")
    st.caption("Validation permutation importance measures influence on model error, not billing causality.")

else:
    st.title("Method and setup")
    st.markdown("""
    Predict one month ahead **after the current month's bill is finalized**. One shared model produces an individual forecast for each customer.

    1. Step 1 checks the data, builds history features and prepares chronological splits.
    2. Step 2 selects the model on validation months and refits it with pre-test labels.
    3. Step 3 adds estimated ranges, customer explanations and review priorities.
    4. This dashboard reads those outputs and can run the saved model for an existing latest-month customer.

    The customer selector uses customers already in the dataset. New invoices must pass through the same feature preparation before forecasts are refreshed.
    """)
    st.subheader("High-bill policy")
    st.code("threshold = recent_mean + max(20% × recent_mean, 50 bill units, 2 × recent_std)")
    st.write("A point alert means the prediction exceeds the threshold. The wider watchlist means the estimated upper bill range exceeds it. Require at least three recorded bills for automatic alerts.")
    st.subheader("Update after running your notebooks")
    st.code("python sync_dashboard_inputs.py\npython -m streamlit run app.py", language="bash")
    st.write("Keep app.py, sync_dashboard_inputs.py, model_outputs, monitoring_outputs and dashboard_inputs together. The supplied package includes ready-to-view demo outputs.")
    st.subheader("Model assumptions and limitations")
    for limitation in monitoring_summary["limitations"]:
        st.write("• " + limitation)
    st.caption("Source: the supplied synthetic dataset and executed Steps 1–3. No live telecom data connection is configured.")
