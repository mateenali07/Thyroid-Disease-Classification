from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="ThyroAI | Thyroid Profile Classification",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Styling
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1000px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(110deg, #153B60, #117D89);
            padding: 30px;
            border-radius: 18px;
            margin-bottom: 24px;
        }

        .hero h1 {
            color: white !important;
            margin: 0;
            padding: 0;
        }

        .hero p {
            color: #E0F2F5;
            margin-bottom: 0;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-radius: 12px;
            padding: 15px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Load saved model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


model_path = Path(__file__).resolve().parent / "thyroid_disease.pkl"

if not model_path.exists():
    st.error(
        "Model file not found. Place thyroid_disease.pkl "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(model_path)
except Exception:
    st.error(
        "Unable to load the model. Check the model file "
        "and installed package versions."
    )
    st.stop()


features = ["Age", "Gender", "TSH", "T3", "TT4", "FTI"]

if list(getattr(model, "feature_names_in_", [])) != features:
    st.error(
        "The model does not match the expected six input features. "
        "Use the model exported from your updated notebook."
    )
    st.stop()


# Header
st.markdown(
    """
    <div class="hero">
        <h1>🩺 ThyroAI</h1>
        <p>Thyroid Profile Classification</p>
        <p>Powered by Gradient Boosting</p>
    </div>
    """,
    unsafe_allow_html=True
)


# Model information
c1, c2, c3 = st.columns(3)

c1.metric("Algorithm", "Gradient Boosting")
c2.metric("Input Features", "6")
c3.metric("Model Status", "Ready")


with st.expander("About the Model"):
    st.write(
        "ThyroAI analyzes patient information and laboratory measurements "
        "to predict whether a diagnosis code is present or absent "
        "in the source dataset."
    )

    st.caption(
        "The model does not distinguish individual thyroid conditions. "
        "Predictions are intended for research and demonstration purposes."
    )

    st.caption(
        "Laboratory units are not specified in the original dataset "
        "documentation. Confirm compatible units and scales before "
        "entering measurements."
    )

    st.caption(
        "Missing laboratory measurements are filled using medians "
        "learned from the training data."
    )


# Patient form
# Enter does not submit the form.
# Use Tab to move between fields and the button to run analysis.
with st.form("patient_form", enter_to_submit=True):

    st.subheader("Patient Information")

    left, right = st.columns(2)

    with left:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=110,
            value=22,
            step=1
        )

    with right:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    st.divider()

    st.subheader("Laboratory Results")

    st.caption(
        "Enter values from your laboratory report. "
        "Leave unavailable tests blank. "
        "Press Tab to move to the next field."
    )

    left, right = st.columns(2)

    with left:
        tsh = st.text_input(
            "Thyroid-Stimulating Hormone (TSH)",
            placeholder="Enter TSH value",
            help=(
                "A hormone that signals the thyroid gland "
                "to produce thyroid hormones."
            )
        )

        t3 = st.text_input(
            "Triiodothyronine (T3)",
            placeholder="Enter T3 value",
            help="One of the hormones produced by the thyroid gland."
        )

    with right:
        tt4 = st.text_input(
            "Total Thyroxine (TT4)",
            placeholder="Enter TT4 value",
            help="A measurement of total T4 hormone in the blood."
        )

        fti = st.text_input(
            "Free Thyroxine Index (FTI)",
            placeholder="Enter FTI value",
            help=(
                "A calculated index used to assess "
                "thyroid hormone levels."
            )
        )

    st.write("")

    submitted = st.form_submit_button(
        "Analyze Thyroid Profile",
        type="primary",
        use_container_width=True
    )


# Validate inputs and predict
if submitted:

    lab_inputs = {
        "TSH": tsh,
        "T3": t3,
        "TT4": tt4,
        "FTI": fti
    }

    patient = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0
    }

    missing_tests = []

    try:
        for name, text in lab_inputs.items():

            text = text.strip()

            if text == "":
                patient[name] = np.nan
                missing_tests.append(name)
                continue

            try:
                value = float(text)
            except ValueError:
                raise ValueError(
                    f"Enter a valid number for {name}."
                ) from None

            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be a finite, non-negative number."
                )

            patient[name] = value

        if len(missing_tests) == len(lab_inputs):
            raise ValueError(
                "Enter at least one laboratory measurement "
                "before running the analysis."
            )

    except ValueError as error:
        st.error(str(error))

    else:
        inp = pd.DataFrame([patient], columns=features)

        try:
            prediction = int(model.predict(inp)[0])
        except Exception:
            st.error(
                "Prediction could not be completed. "
                "Check the model and its preprocessing pipeline."
            )

        else:
            st.divider()
            st.subheader("Analysis Result")

            if prediction == 1:
                st.info("Diagnosis code predicted: Present")
            else:
                st.info("Diagnosis code predicted: Absent")

            st.caption(
                f"Laboratory measurements provided: "
                f"{len(lab_inputs) - len(missing_tests)} of 4"
            )

            if missing_tests:
                st.warning(
                    "Missing measurements were filled using training "
                    "medians: "
                    + ", ".join(missing_tests)
                    + ". Predictions with missing measurements "
                    "may be less informative."
                )

            st.caption(
                "This result represents a dataset diagnosis-code category. "
                "It does not confirm or rule out thyroid disease."
            )


# Footer
st.divider()
st.caption("ThyroAI • Python · Scikit-learn · Streamlit")