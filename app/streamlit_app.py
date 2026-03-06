import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Bulblet Yield Prediction System", page_icon="🌷", layout="wide")

st.title("🌷 Bulblet Yield Prediction System")

BASE_DIR = Path(".").resolve()
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_DIR / "model_dozajli_v2.pkl")
    le_species = joblib.load(MODEL_DIR / "label_encoder_tur_dozajli_v2.pkl")
    le_application = joblib.load(MODEL_DIR / "label_encoder_uyg_dozajli_v2.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler_dozajli_v2.pkl")
    return model, le_species, le_application, scaler

model, le_species, le_application, scaler = load_resources()

application_display_map = {
    "kontrol": "Control",
    "2ye_bolme": "Division into Two",
    "4e_bolme": "Division into Four",
    "mikoriza": "Mycorrhiza",
    "bakteri": "Bacterial Application"
}

reverse_application_map = {v: k for k, v in application_display_map.items()}

small_species = [
    "Tulipa cinnabarina K.perss.",
    "Tulipa pulchella (Regel) Baker",
    "Tulipa biflora Pall",
    "Tulipa koyuncui Eker"
]

optimum_mycorrhiza = 50
optimum_bacteria = 50

def dose_effect_factor(dose, optimum):
    deviation = abs(dose - optimum)
    if deviation == 0:
        return 1.0
    elif deviation <= 25:
        return 0.9
    elif deviation <= 50:
        return 0.75
    else:
        return 0.6

def predict(species, application_tr, circumference, weight, mycorrhiza, bacteria):

    species_enc = le_species.transform([species])[0]
    application_enc = le_application.transform([application_tr])[0]

    X = np.array([[species_enc, application_enc, circumference, weight, mycorrhiza, bacteria]])

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    bulblet_number = float(prediction[0])
    bulblet_weight = float(prediction[1])

    bulblet_number = np.clip(bulblet_number, 1, 3)

    if species in small_species:
        bulblet_weight = np.clip(bulblet_weight, 0.1, 5.0)
    else:
        bulblet_weight = max(bulblet_weight, 0.1)

    myco_factor = dose_effect_factor(mycorrhiza, optimum_mycorrhiza)
    bact_factor = dose_effect_factor(bacteria, optimum_bacteria)

    combined_factor = (myco_factor + bact_factor) / 2

    bulblet_number = bulblet_number * combined_factor
    bulblet_weight = bulblet_weight * combined_factor

    return bulblet_number, bulblet_weight

col1, col2 = st.columns([1,2])

with col1:

    st.header("Input Parameters")

    species = st.selectbox("Tulip Species", list(le_species.classes_))

    application_display = st.selectbox("Application Type", list(application_display_map.values()))

    application_tr = reverse_application_map[application_display]

    circumference = st.slider("Initial Bulb Circumference (mm)", 5.0, 50.0, 20.0)

    weight = st.number_input("Initial Bulb Weight (g)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)

    mycorrhiza = st.slider("Mycorrhiza Dose (ml)", 0, 200, 50)

    bacteria = st.slider("Bacteria Dose (ml)", 0, 200, 50)

    run_prediction = st.button("Run Prediction")

with col2:

    st.header("Prediction Results")

    if run_prediction:

        n, w = predict(species, application_tr, circumference, weight, mycorrhiza, bacteria)

        st.success(f"Predicted Number of Bulblets: {n:.2f}")
        st.success(f"Predicted Bulblet Weight: {w:.2f} g")

        comparison_data = []

        for app_tr in le_application.classes_:
            n2, w2 = predict(species, app_tr, circumference, weight, mycorrhiza, bacteria)
            comparison_data.append([application_display_map[app_tr], n2, w2])

        df = pd.DataFrame(comparison_data, columns=["Application Type","Bulblet Number","Bulblet Weight (g)"])

        st.subheader("Comparison Across Application Types")

        st.dataframe(df)

        fig = px.bar(
            df,
            x="Application Type",
            y=["Bulblet Number","Bulblet Weight (g)"],
            barmode="group",
            title="Predicted Bulblet Yield by Application Type"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Decision Support Recommendations")

        if mycorrhiza < optimum_mycorrhiza:
            st.info("Mycorrhiza dose is below optimum (50 ml). Increasing dose may improve yield.")
        elif mycorrhiza > optimum_mycorrhiza + 50:
            st.warning("Mycorrhiza dose is excessively high and may reduce yield.")

        if bacteria < optimum_bacteria:
            st.info("Bacteria dose is below optimum (50 ml). Increasing dose may improve yield.")
        elif bacteria > optimum_bacteria + 50:
            st.warning("Bacteria dose is excessively high and may reduce yield.")

    else:
        st.info("Please enter input parameters and click 'Run Prediction'.")

st.markdown("---")

st.caption("Machine learning based prediction system for tulip bulblet production.")
