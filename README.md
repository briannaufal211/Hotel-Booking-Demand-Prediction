# 📊 Hotel Booking Cancellation Prediction  
**Capstone Project – Machine Learning Classification**

## 📌 Project Overview
Pembatalan pemesanan hotel (*booking cancellation*) merupakan permasalahan utama dalam industri perhotelan karena berdampak langsung pada tingkat okupansi, pendapatan, dan perencanaan operasional.  
Project ini bertujuan untuk membangun **model Machine Learning klasifikasi** yang mampu **memprediksi apakah sebuah booking akan dibatalkan atau tidak** berdasarkan karakteristik pelanggan dan detail pemesanan.

Model yang dihasilkan diharapkan dapat membantu pihak hotel dalam:
- Mengidentifikasi booking berisiko tinggi untuk dibatalkan
- Menyusun strategi overbooking yang lebih optimal
- Meningkatkan pendapatan dan efisiensi operasional

---

## 🎯 Business Problem
Hotel sering mengalami:
- Tingginya tingkat pembatalan pemesanan
- Kehilangan potensi pendapatan
- Ketidaktepatan strategi overbooking

Tanpa prediksi yang akurat, hotel sulit membedakan booking yang kemungkinan besar akan dibatalkan dan yang tidak.

---

## 🎯 Objective
Membangun model Machine Learning untuk:
- Memprediksi **status pembatalan booking (`is_canceled`)**
- Meminimalkan booking berisiko tinggi yang tidak terdeteksi
- Mendukung pengambilan keputusan bisnis berbasis data

---

## 🧾 Dataset Information
Dataset yang digunakan adalah **Hotel Booking Demand Dataset**, dengan **11 fitur utama**:

| Feature | Type | Description |
|------|------|------|
| country | Categorical | Negara asal tamu |
| market_segment | Categorical | Sumber pemesanan |
| previous_cancellations | Numerical | Jumlah pembatalan sebelumnya |
| booking_changes | Numerical | Jumlah perubahan booking |
| deposit_type | Categorical | Jenis deposit |
| days_in_waiting_list | Numerical | Lama di waiting list |
| customer_type | Categorical | Tipe pelanggan |
| reserved_room_type | Categorical | Tipe kamar yang dipesan |
| required_car_parking_spaces | Numerical | Kebutuhan parkir |
| total_of_special_requests | Numerical | Jumlah permintaan khusus |
| is_canceled | Binary | **Target variable (1 = Cancel, 0 = Not Cancel)** |

---

## 🧠 Analytical Approach
Project ini menggunakan pendekatan **Supervised Learning – Binary Classification**, dengan tahapan:

1. Business Understanding  
2. Data Understanding  
3. Data Cleaning  
4. Exploratory Data Analysis (EDA)  
5. Data Preparation  
6. Modeling & Benchmarking  
7. Hyperparameter Tuning  
8. Model Evaluation  
9. Feature Importance & Interpretability (SHAP)  
10. Error Analysis  
11. Business Insight & Interpretation  
12. Conclusion & Recommendation  
13. Model Saving & Deployment Preparation  

---

## ⚙️ Machine Learning Pipeline
Pipeline dibangun secara end-to-end menggunakan `Pipeline` dan `ColumnTransformer`:

- **Numerical Features**
  - Imputation (Median)
  - Standard Scaling

- **Categorical Features**
  - Imputation (Most Frequent)
  - One-Hot Encoding

- **Imbalanced Handling**
  - RandomOverSampler

- **Models Evaluated**
  - Logistic Regression (Baseline)
  - Decision Tree
  - Random Forest
  - XGBoost
  - LightGBM

---

## 📐 Evaluation Metrics
Karena dataset **imbalanced**, metrik evaluasi difokuskan pada:

### 🔹 Primary Metrics
- **Recall (is_canceled = 1)**
- **F1-Score**

### 🔹 Supporting Metrics
- Precision
- Accuracy
- ROC-AUC

**Alasan utama:**  
Lebih penting mendeteksi booking yang benar-benar akan dibatalkan (minimize False Negative) dibandingkan hanya mengejar akurasi tinggi.

---

## 📈 Key Results
- Model terbaik menunjukkan performa optimal pada **Recall dan F1-Score**
- Model mampu membedakan booking berisiko tinggi dengan baik
- Fitur-fitur penting meliputi:
  - Deposit Type
  - Previous Cancellations
  - Market Segment
  - Total Special Requests

---

## 🔍 Model Interpretability
Project ini menggunakan:
- **Feature Importance**
- **SHAP (SHapley Additive exPlanations)**

Untuk memastikan:
- Model dapat dijelaskan secara bisnis
- Keputusan model transparan
- Tidak hanya akurat tetapi juga interpretable

---

## 💡 Business Insight
Beberapa insight utama:
- Booking dengan **Non Refund Deposit** memiliki kecenderungan lebih kecil untuk dibatalkan
- Pelanggan dengan riwayat pembatalan tinggi berisiko besar melakukan cancel ulang
- Semakin banyak permintaan khusus, semakin kecil kemungkinan pembatalan

---

## 📌 Recommendation
- Terapkan model ini sebagai **early warning system**
- Fokuskan strategi overbooking pada booking berisiko tinggi
- Gunakan output probabilitas untuk segmentasi pelanggan
- Integrasikan model ke sistem reservasi hotel

---

## 💾 Model Deployment
Model akhir disimpan menggunakan **Pickle**, mencakup:
- Preprocessing
- Oversampling
- Classifier

Sehingga:
- Data baru bisa langsung diprediksi
- Tidak perlu preprocessing manual ulang
- Siap untuk production / API deployment

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost, LightGBM
- SHAP
- Matplotlib, Seaborn

---

## 👨‍🎓 Author
**Brian Naufal**  
Data & Machine Learning Enthusiast  
