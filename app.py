import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Sayfa ayarı
st.set_page_config(
    page_title="Kalp Yetmezliği Risk Tahmin Sistemi",
    page_icon="❤️",
    layout="wide"
)



# Model yükleme
model = joblib.load("C:/Users/team/OneDrive/Masaüstü/projelerim/proje_1/heart+failure+clinical+records/rf_final.pkl")

st.title("❤️ Kalp Yetmezliği Risk Tahmin Sistemi")

st.markdown("""
Bu uygulama kalp yetmezliği hastalarında ölüm riskini
makine öğrenmesi kullanarak tahmin etmektedir.
""")


# Sidebar

st.sidebar.header("Hasta Bilgileri")

age = st.sidebar.number_input("Yaş", 40, 100, 60)

anaemia = st.sidebar.selectbox(
    "Anemi",
    ["Hayır", "Evet"]
)
anaemia = 0 if anaemia == "Hayır" else 1

creatinine_phosphokinase = st.sidebar.number_input(
    "Kreatin Fosfokinaz (CPK)", 20, 8000, 250
)

diabetes = st.sidebar.selectbox(
    "Diyabet",
    ["Hayır", "Evet"]
)
diabetes = 0 if diabetes == "Hayır" else 1

ejection_fraction = st.sidebar.number_input(
    "Ejeksiyon Fraksiyonu (%)", 10, 80, 38
)

high_blood_pressure = st.sidebar.selectbox(
    "Hipertansiyon",
    ["Hayır", "Evet"]
)
high_blood_pressure = 0 if high_blood_pressure == "Hayır" else 1

platelets = st.sidebar.number_input(
    "Trombosit (×10³/µL)", 20.0, 900.0, 260.0
)

serum_creatinine = st.sidebar.number_input(
    "Serum Kreatinin (mg/dL)", 0.4, 10.0, 1.2
)

serum_sodium = st.sidebar.number_input(
    "Serum Sodyum (mEq/L)", 110, 150, 137
)

sex = st.sidebar.selectbox(
    "Cinsiyet",
    ["Kadın", "Erkek"]
)
sex = 0 if sex == "Kadın" else 1

smoking = st.sidebar.selectbox(
    "Sigara Kullanımı",
    ["Hayır", "Evet"]
)
smoking = 0 if smoking == "Hayır" else 1

# Feature Engineering

new_low_ef = 1 if ejection_fraction <= 40 else 0

new_severe_hf = 1 if ejection_fraction < 35 else 0

new_hyponatremia = 1 if serum_sodium < 135 else 0

new_renal_risk = 1 if serum_creatinine > 1.5 else 0

new_elderly = 1 if age >= 65 else 0

new_creat_sodium_ratio = serum_creatinine / serum_sodium

new_age_ef_ratio = age / ejection_fraction

new_ht_dm = high_blood_pressure * diabetes

new_cardio_renal = new_severe_hf * new_renal_risk

new_age_creatinine = age * serum_creatinine

new_age_ef = age / ejection_fraction

new_comorbidity_score = (
    anaemia +
    diabetes +
    high_blood_pressure
)

new_clinical_risk_index = (
    new_elderly +
    new_renal_risk +
    new_severe_hf +
    new_hyponatremia
)


# Tahmin DataFrame

input_df = pd.DataFrame({

    "age":[age],
    "anaemia":[anaemia],
    "creatinine_phosphokinase":[creatinine_phosphokinase],
    "diabetes":[diabetes],
    "ejection_fraction":[ejection_fraction],
    "high_blood_pressure":[high_blood_pressure],
    "platelets":[platelets],
    "serum_creatinine":[serum_creatinine],
    "serum_sodium":[serum_sodium],
    "sex":[sex],
    "smoking":[smoking],

    "NEW_LOW_EF":[new_low_ef],
    "NEW_SEVERE_HF":[new_severe_hf],
    "NEW_HYPONATREMIA":[new_hyponatremia],
    "NEW_RENAL_RISK":[new_renal_risk],
    "NEW_ELDERLY":[new_elderly],
    "NEW_CREAT_SODIUM_RATIO":[new_creat_sodium_ratio],
    "NEW_AGE_EF_RATIO":[new_age_ef_ratio],
    "NEW_HT_DM":[new_ht_dm],
    "NEW_CARDIO_RENAL":[new_cardio_renal],
    "NEW_AGE_CREATININE":[new_age_creatinine],
    "NEW_AGE_EF":[new_age_ef],
    "NEW_COMORBIDITY_SCORE":[new_comorbidity_score],
    "NEW_CLINICAL_RISK_INDEX":[new_clinical_risk_index]
})

# Tahmin

if st.button("Risk Tahmini Yap"):

    risk = model.predict_proba(input_df)[0][1]

    st.subheader("Tahmin Sonucu")

    st.metric(
        "Tahmini Mortalite Riski",
        f"{risk*100:.1f}%"
    )

    st.progress(float(risk))

    if risk < 0.30:

        st.success("🟢 DÜŞÜK RİSK")

        st.info("""
       Tahmini mortalite riski düşüktür.

        Hastanın klinik profili kısa dönem sağkalım açısından
        olumlu görünmektedir.

        Rutin klinik takip ve standart tedavinin
        sürdürülmesi önerilir.
        """)

    elif risk < 0.60:

        st.warning("🟡 ORTA RİSK")

        st.info("""
       Tahmini mortalite riski orta düzeydedir.

        Hastada prognozu olumsuz etkileyebilecek
        bazı klinik risk faktörleri bulunmaktadır.

        Daha yakın klinik takip ve tedavinin
        optimize edilmesi önerilir.
        """)

    else:

        st.error("🔴 YÜKSEK RİSK")

        st.info("""
        Tahmini mortalite riski yüksektir.

        Model, artmış mortalite ile ilişkili
        risk faktörlerini belirlemiştir.
        
        Ayrıntılı klinik değerlendirme,
        yakın takip ve uygun tedavi planlaması
        önerilmektedir.
        """)

    st.write(f"Model Olasılık Skoru {risk:.3f}")




# Proje Bilgisi

with st.expander("Proje Hakkında"):
    st.markdown("""
    ### Amaç

    Bu uygulama, kalp yetmezliği hastalarında mortalite riskini
    makine öğrenmesi yöntemleri kullanarak tahmin etmek amacıyla geliştirilmiştir.
    
    ### Kullanılan Model
    
    - Random Forest
    - 5 Katlı Çapraz Doğrulama (5-Fold Cross Validation)
    - Özellik Mühendisliği (Feature Engineering)
    - SHAP ile Model Açıklanabilirliği
    
    ### Geliştirici
    
    Bu proje veri bilimi ve klinik araştırmalar alanındaki
    uygulamaları göstermek amacıyla hazırlanmıştır.
    
    ⚠️ **Uyarı:** Bu uygulama yalnızca eğitim ve araştırma amaçlıdır.
    Klinik karar verme sürecinin yerine geçmez.



    Geliştiren:
    Menekşe Ayhan
    """)

st.markdown("---")

st.caption(
    "© 2026 | Kalp Yetmezliği Risk Tahmin Sistemi | "
    "Makine Öğrenmesi ile Klinik Karar Destek Uygulaması"
)