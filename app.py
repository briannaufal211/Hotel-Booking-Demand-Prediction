# app.py

import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL (Pipeline)
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model_hotel_booking.pkl")

model = load_model()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hotel Cancellation Prediction",
    layout="wide",
    page_icon="🏨"
)

st.title("🏨 Hotel Booking Cancellation Prediction")
st.markdown("Predict the likelihood of booking cancellation and support business decision-making")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("🔧 Input Booking Data")

lead_time = st.sidebar.number_input("Lead Time", 0, 500, 50)
previous_cancellations = st.sidebar.number_input("Previous Cancellations", 0, 10, 0)
booking_changes = st.sidebar.number_input("Booking Changes", 0, 10, 0)

deposit_type = st.sidebar.selectbox(
    "Deposit Type",
    ["No Deposit", "Non Refund", "Refundable"]
)

market_segment = st.sidebar.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate"]
)

customer_type = st.sidebar.selectbox(
    "Customer Type",
    ["Transient", "Contract", "Group", "Transient-Party"]
)

# =========================
# SINGLE PREDICTION
# =========================
st.subheader("🔍 Single Prediction")

if st.sidebar.button("🚀 Predict"):

    input_data = pd.DataFrame({
        "lead_time": [lead_time],
        "previous_cancellations": [previous_cancellations],
        "booking_changes": [booking_changes],
        "deposit_type": [deposit_type],
        "market_segment": [market_segment],
        "customer_type": [customer_type]
    })

    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", "Cancel" if prediction == 1 else "Not Cancel")

        with col2:
            st.metric("Probability", f"{prob:.2%}")

        with col3:
            st.metric("Risk Level", 
                      "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low")

        st.progress(float(min(max(prob, 0), 1)))

        # =========================
        # DECISION SUPPORT
        # =========================
        st.subheader("💡 Business Recommendation")

        if prob > 0.7:
            st.error("⚠️ High Risk → Require Deposit or Prepayment")
        elif prob > 0.4:
            st.warning("⚠️ Medium Risk → Send Reminder / Offer Incentive")
        else:
            st.success("✅ Low Risk → No Action Needed")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# =========================
# BATCH PREDICTION
# =========================
st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview Data:")
    st.dataframe(df.head())

    expected_columns = [
        "lead_time", "previous_cancellations", "booking_changes",
        "deposit_type", "market_segment", "customer_type"
    ]

    if not all(col in df.columns for col in expected_columns):
        st.error("❌ CSV format tidak sesuai. Pastikan kolom sesuai dengan input model.")
    else:
        try:
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]

            df["Prediction"] = predictions
            df["Probability"] = probabilities
            df["Risk Level"] = df["Probability"].apply(
                lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
            )

            st.write("✅ Prediction Result:")
            st.dataframe(df)

            # =========================
            # VISUALIZATION
            # =========================
            st.subheader("📊 Prediction Distribution")
            st.bar_chart(df["Risk Level"].value_counts())

            # =========================
            # DOWNLOAD
            # =========================
            st.download_button(
                label="📥 Download Result",
                data=df.to_csv(index=False),
                file_name="prediction_result.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed for Hotel Revenue Optimization using Machine Learning")