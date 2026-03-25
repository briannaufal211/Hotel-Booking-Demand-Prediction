# app.py

import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hotel Cancellation Prediction",
    layout="wide",
    page_icon="🏨"
)

import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
import lzma
import joblib

@st.cache_resource
def load_model():
    with lzma.open("model_hotel_booking1.xz", "rb") as f:
        return joblib.load(f)

model = load_model()

# ambil kolom model (penting untuk get_dummies)
model_columns = [
    "lead_time",
    "previous_cancellations",
    "booking_changes",
    "deposit_type_No Deposit",
    "deposit_type_Non Refund",
    "deposit_type_Refundable",
    "market_segment_Online TA",
    "market_segment_Offline TA/TO",
    "market_segment_Direct",
    "market_segment_Corporate",
    "customer_type_Transient",
    "customer_type_Contract",
    "customer_type_Group",
    "customer_type_Transient-Party"
]

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_input(df_input):
    df_encoded = pd.get_dummies(df_input)

    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[model_columns]
    return df_encoded

# =========================
# HEADER
# =========================
st.title("🏨 Hotel Booking Cancellation Prediction")
st.markdown("### Decision Support System for Revenue Optimization")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔧 Booking Input")

lead_time = st.sidebar.number_input("Lead Time", 0, 500, 50)
previous_cancellations = st.sidebar.number_input("Previous Cancellations", 0, 10, 0)
booking_changes = st.sidebar.number_input("Booking Changes", 0, 10, 0)

deposit_type = st.sidebar.selectbox("Deposit Type",
    ["No Deposit", "Non Refund", "Refundable"])

market_segment = st.sidebar.selectbox("Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate"])

customer_type = st.sidebar.selectbox("Customer Type",
    ["Transient", "Contract", "Group", "Transient-Party"])

st.sidebar.markdown("---")
st.sidebar.markdown("## 📌 Model Info")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Encoding: pd.get_dummies")

# =========================
# SINGLE PREDICTION
# =========================
st.subheader("🔍 Single Prediction")

if st.sidebar.button("🚀 Predict"):

    input_df = pd.DataFrame({
        "lead_time": [lead_time],
        "previous_cancellations": [previous_cancellations],
        "booking_changes": [booking_changes],
        "deposit_type": [deposit_type],
        "market_segment": [market_segment],
        "customer_type": [customer_type]
    })

    try:
        processed = preprocess_input(input_df)

        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction", "Cancel" if pred else "Not Cancel")
        col2.metric("Probability", f"{prob:.2%}")
        col3.metric("Risk Level",
            "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low")

        st.progress(prob)

        st.subheader("💡 Recommendation")

        if prob > 0.7:
            st.error("High Risk → Require Deposit / Prepayment")
        elif prob > 0.4:
            st.warning("Medium Risk → Send Reminder / Offer Incentive")
        else:
            st.success("Low Risk → No Action Needed")

    except Exception as e:
        st.error(f"Error: {e}")

# =========================
# BATCH PREDICTION
# =========================
st.subheader("📂 Batch Prediction")

# =========================
# SAMPLE CSV DOWNLOAD
# =========================
st.markdown("### 📥 Download Sample CSV")

sample_df = pd.DataFrame({
    "lead_time": [50],
    "previous_cancellations": [0],
    "booking_changes": [1],
    "deposit_type": ["No Deposit"],
    "market_segment": ["Online TA"],
    "customer_type": ["Transient"]
})

st.download_button(
    label="Download Sample CSV",
    data=sample_df.to_csv(index=False),
    file_name="sample_hotel_booking.csv",
    mime="text/csv"
)

st.markdown("Upload file dengan format yang sama seperti sample di atas 👇")

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.write("Preview Data")
    st.dataframe(df.head())

    required_cols = [
        "lead_time", "previous_cancellations", "booking_changes",
        "deposit_type", "market_segment", "customer_type"
    ]

    # isi kolom missing
    for col in required_cols:
        if col not in df.columns:
            if col in ["lead_time", "previous_cancellations", "booking_changes"]:
                df[col] = 0
            else:
                df[col] = "Unknown"

    df_model = df[required_cols]

    try:
        processed = preprocess_input(df_model)

        pred = model.predict(processed)
        prob = model.predict_proba(processed)[:, 1]

        df["Prediction"] = pred
        df["Probability"] = prob
        df["Risk Level"] = df["Probability"].apply(
            lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
        )

        st.success(f"Prediction success! {len(df)} records processed")

        # =========================
        # KPI
        # =========================
        st.subheader("📊 Business Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Booking", len(df))
        c2.metric("High Risk", (df["Risk Level"] == "High").sum())
        c3.metric("Avg Probability", f"{df['Probability'].mean():.2%}")

        # =========================
        # BUSINESS IMPACT
        # =========================
        st.subheader("💰 Business Impact")

        avg_revenue = 100
        loss = df["Probability"].sum() * avg_revenue

        st.metric("Estimated Revenue at Risk", f"${loss:,.0f}")

        # =========================
        # VISUALIZATION
        # =========================
        st.subheader("📊 Risk Distribution")
        st.bar_chart(df["Risk Level"].value_counts())

        # =========================
        # INSIGHT
        # =========================
        st.subheader("🧠 Key Insights")

        high = (df["Risk Level"] == "High").sum()

        st.write(f"""
        - {high} bookings memiliki risiko tinggi
        - Model membantu deteksi cancel lebih awal
        - Intervensi seperti deposit bisa mengurangi risiko
        """)

        # =========================
        # DOWNLOAD
        # =========================
        st.download_button(
            "📥 Download Result",
            df.to_csv(index=False),
            "prediction.csv"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built for Decision Support System (DSS) - Hotel Revenue Optimization")
