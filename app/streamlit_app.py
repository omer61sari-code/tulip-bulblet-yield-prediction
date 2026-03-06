import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Bulblet Yield Prediction System",
    page_icon="🌷",
    layout="wide"
)

st.title("🌷 Bulblet Yield Prediction System")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_resources():

    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"

    model = joblib.load(MODEL_DIR / "model_dozajli_v2.pkl")
    le_species = joblib.load(MODEL_DIR / "label_encoder_tur_dozajli_v2.pkl")
    le_application = joblib.load(MODEL_DIR / "label_encoder_uyg_dozajli_v2.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler_dozajli_v2.pkl")

    return model, le_species, le_application, scaler


model, le_species, le_application, scaler = load_resources()

# -------------------------------------------------
# APPLICATION MAP
# -------------------------------------------------

application_display_map = {
    "kontrol": "Control",
    "2ye_bolme": "Division into Two",
    "4e_bolme": "Division into Four",
    "mikoriza": "Mycorrhiza",
    "bakteri": "Bacterial Application"
}

reverse_application_map = {v: k for k, v in application_display_map.items()}

# -------------------------------------------------
# SMALL SPECIES
# -------------------------------------------------

small_species = [
    "Tulipa cinnabarina K.perss.",
    "Tulipa pulchella (Regel) Baker",
    "Tulipa biflora Pall",
    "Tulipa koyuncui Eker"
]

# -------------------------------------------------
# OPTIMUM DOSES
# -------------------------------------------------

optimum_mycorrhiza = 50
optimum_bacteria = 50

# -------------------------------------------------
# DOSE RESPONSE
# -------------------------------------------------

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

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------

def predict(species, application_tr, circumference, weight, mycorrhiza, bacteria):

    species_enc = le_species.transform([species])[0]
    application_enc = le_application.transform([application_tr])[0]

    X = np.array([[species_enc,
                   application_enc,
                   circumference,
                   weight,
                   mycorrhiza,
                   bacteria]])

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    bulblet_number = float(prediction[0])
    bulblet_weight = float(prediction[1])

    # -----------------------------
    # DOSE EFFECT
    # -----------------------------

    myco_factor = dose_effect_factor(mycorrhiza, optimum_mycorrhiza)
    bact_factor = dose_effect_factor(bacteria, optimum_bacteria)

    dose_factor = (myco_factor + bact_factor) / 2

    bulblet_number *= dose_factor
    bulblet_weight *= dose_factor

    # -----------------------------
    # BULB SIZE EFFECT
    # -----------------------------

    size_factor = 1 + ((circumference - 20) / 80)
    weight_factor = 1 + ((weight - 10) / 80)

    bulblet_number *= size_factor * weight_factor
    bulblet_weight *= size_factor * weight_factor

    # -----------------------------
    # APPLICATION EFFECT
    # -----------------------------

    application_effects = {
        "kontrol": 1.0,
        "2ye_bolme": 1.15,
        "4e_bolme": 1.25,
        "mikoriza": 1.10,
        "bakteri": 1.08
    }

    bulblet_number *= application_effects.get(application_tr, 1)

    # -----------------------------
    # BIOLOGICAL LIMITS
    # -----------------------------

    bulblet_number = np.clip(bulblet_number, 1, 4)

    if species in small_species:
        bulblet_weight = np.clip(bulblet_weight, 0.1, 5)
    else:
        bulblet_weight = np.clip(bulblet_weight, 0.1, 20)

    return np.array([bulblet_number, bulblet_weight])


# -------------------------------------------------
# USER INTERFACE
# -------------------------------------------------

col1, col2 = st.columns([1,2])

with col1:

    st.header("Input Parameters")

    species = st.selectbox(
        "Tulip Species",
        list(le_species.classes_)
    )

    application_display = st.selectbox(
        "Application Type",
        list(application_display_map.values())
    )

    application_tr = reverse_application_map[application_display]

    circumference = st.slider(
        "Initial Bulb Circumference (mm)",
        5.0, 50.0, 20.0
    )

    weight = st.number_input(
        "Initial Bulb Weight (g)",
        0.1, 100.0, 10.0
    )

    mycorrhiza = st.slider(
        "Mycorrhiza Dose (ml)",
        0, 200, 50
    )

    bacteria = st.slider(
        "Bacteria Dose (ml)",
        0, 200, 50
    )

    run_prediction = st.button("Run Prediction")

# -------------------------------------------------
# RESULTS
# -------------------------------------------------

with col2:

    st.header("Prediction Results")

    if run_prediction:

        n,w = predict(
            species,
            application_tr,
            circumference,
            weight,
            mycorrhiza,
            bacteria
        )

        st.success(f"Predicted Number of Bulblets: {n:.2f}")
        st.success(f"Predicted Bulblet Weight: {w:.2f} g")

        comparison_data = []

        for app_tr in le_application.classes_:

            n2,w2 = predict(
                species,
                app_tr,
                circumference,
                weight,
                mycorrhiza,
                bacteria
            )

            comparison_data.append([
                application_display_map[app_tr],
                round(n2,2),
                round(w2,2)
            ])

        df = pd.DataFrame(
            comparison_data,
            columns=[
                "Application Type",
                "Bulblet Number",
                "Bulblet Weight (g)"
            ]
        )

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

    else:

        st.info("Please enter the input parameters and click Run Prediction")

# -------------------------------------------------

st.markdown("---")

st.caption(
"Machine learning based prediction system for tulip bulblet production."
)
