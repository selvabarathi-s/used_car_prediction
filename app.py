
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "artifacts" / "models" / "model_metadata.json"
SRC_DIR = BASE_DIR / "src"

if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

try:
    # Required so joblib can resolve the custom BrandTargetEncoder
    from car_transformers import BrandTargetEncoder  # noqa: F401
except Exception:
    pass


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.3rem;
        }
        .price-card {
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(100,116,139,.20);
            background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(16,185,129,.08));
            margin: 0.8rem 0 1rem;
        }
        .price-label {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.25rem;
        }
        .price-value {
            font-size: 2.5rem;
            font-weight: 850;
            letter-spacing: -0.03em;
        }
        .muted {
            color: #64748b;
            font-size: 0.9rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 750;
            margin: 0.4rem 0 0.8rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load model + metadata
# ============================================================

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}\n\n"
            "Run the notebook training section first."
        )

    model = joblib.load(MODEL_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return model, metadata


try:
    model, metadata = load_assets()
except Exception as exc:
    st.error("Unable to load the trained model.")
    st.code(str(exc))
    st.stop()


# ============================================================
# Metadata helpers
# ============================================================

feature_columns = metadata.get("model_input", {}).get("feature_columns", [])
feature_prep = metadata.get("feature_preparation", {})

label_encoder_classes = feature_prep.get("label_encoder_classes", {})
age_edges = feature_prep.get("vehicle_age_bin_edges", [])
mileage_edges = feature_prep.get("mileage_bin_edges", [])
reference_year = int(feature_prep.get("vehicle_age_reference_year", 2025))

model_name = metadata.get("model_name", "Best Model")
metrics = metadata.get("metrics", {})
mlflow_info = metadata.get("mlflow", {})

if not feature_columns:
    st.error(
        "model_metadata.json does not contain the expected feature schema. "
        "Retrain/save the notebook model so metadata is generated."
    )
    st.stop()

if len(age_edges) != 5 or len(mileage_edges) != 5:
    st.error(
        "Model metadata is missing valid age/mileage bin edges. "
        "The app cannot reproduce the notebook preprocessing safely."
    )
    st.stop()


def get_classes(key, fallback):
    classes = label_encoder_classes.get(key)
    return [str(v) for v in classes] if classes else fallback


fuel_options = get_classes(
    "fuel_type",
    ["GASOLINE", "DIESEL", "HYBRID", "ELECTRIC", "OTHER"],
)

transmission_options = get_classes(
    "transmission",
    ["A/T", "M/T", "CVT", "OTHER"],
)

v_engine_classes = get_classes(
    "is_v_engine",
    ["False", "True"],
)


def encoded_value(value, classes, fallback_name="OTHER"):
    """Reproduce LabelEncoder's integer mapping stored in metadata."""
    classes = [str(c) for c in classes]
    value = str(value)

    if value not in classes:
        if fallback_name in classes:
            value = fallback_name
        else:
            raise ValueError(
                f"'{value}' is not present in the model's learned classes: {classes}"
            )

    return classes.index(value)


def bin_one_hot(value, edges, prefix, labels):
    """
    Reproduce the notebook's:
        pd.qcut(..., labels=..., retbins=True)
        pd.get_dummies(..., drop_first=True, dtype=int)
    """
    clipped = float(np.clip(value, edges[0], edges[-1]))

    binned = pd.cut(
        pd.Series([clipped]),
        bins=edges,
        labels=labels,
        include_lowest=True,
    )

    dummies = pd.get_dummies(
        binned,
        prefix=prefix,
        drop_first=True,
        dtype=int,
    )

    expected = [f"{prefix}_{label}" for label in labels[1:]]

    for col in expected:
        if col not in dummies.columns:
            dummies[col] = 0

    return {col: int(dummies.iloc[0][col]) for col in expected}


def build_features(
    brand,
    model_year,
    mileage,
    horsepower,
    displacement,
    is_v_engine,
    fuel_type,
    transmission,
    accident,
    clean_title,
):
    vehicle_age = reference_year - int(model_year)

    if vehicle_age < 0:
        raise ValueError(
            f"Model year cannot be after the notebook's reference year ({reference_year})."
        )

    mileage_per_year = (
        float(mileage) / vehicle_age if vehicle_age > 0 else float(mileage)
    )

    row = {
        "brand": brand.strip(),
        "fuel_type": encoded_value(fuel_type, fuel_options),
        "transmission": encoded_value(transmission, transmission_options),
        "is_v_engine": encoded_value(is_v_engine, v_engine_classes),
        "clean_title": 1 if clean_title == "Yes" else 0,
        "hp": float(horsepower),
        "engine displacement": float(displacement),
        "Accident_Impact": 1 if accident == "At least 1 accident or damage reported" else 0,
        "Vehicle_Age": float(vehicle_age),
        "Mileage_per_Year": float(mileage_per_year),
    }

    row.update(
        bin_one_hot(
            vehicle_age,
            age_edges,
            "Age",
            ["New", "Mid", "Old", "Very Old"],
        )
    )

    row.update(
        bin_one_hot(
            float(mileage),
            mileage_edges,
            "Milage",
            ["Low", "Medium", "High", "Very High"],
        )
    )

    frame = pd.DataFrame([row])

    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(
            "The generated input is missing model features: "
            + ", ".join(missing)
        )

    # Exact column order used when the model was trained.
    return frame[feature_columns]


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">Used Car Price Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Predict a used car's estimated price using the best trained model "
    "from your MLflow experiment."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar — model information
# ============================================================

with st.sidebar:
    st.header("Model")
    st.success(f"Loaded: {model_name}")

    if "test_rmse" in metrics:
        st.metric("Test RMSE", f"${metrics['test_rmse']:,.0f}")
    if "test_r2" in metrics:
        st.metric("Test R²", f"{metrics['test_r2']:.3f}")
    if "test_mape" in metrics:
        st.metric("Test MAPE", f"{metrics['test_mape']:.2f}%")

    st.divider()

    st.caption("MLflow")
    st.write(f"Experiment: `{mlflow_info.get('experiment_name', 'N/A')}`")
    st.write(f"Run ID: `{mlflow_info.get('run_id', 'N/A')}`")

    tracking_uri = mlflow_info.get("tracking_uri", "")
    if tracking_uri:
        st.link_button("Open MLflow UI", tracking_uri)

    st.divider()
    st.caption(f"Feature count: {len(feature_columns)}")
    st.caption(f"Reference year: {reference_year}")


# ============================================================
# Input form
# ============================================================

st.markdown('<div class="section-title">Vehicle Details</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    brand = st.text_input(
        "Brand",
        value="Ford",
        help="Brand is passed as text; the trained model performs its brand encoding.",
    )

    model_year = st.number_input(
        "Model Year",
        min_value=1980,
        max_value=reference_year,
        value=min(2020, reference_year),
        step=1,
    )

    mileage = st.number_input(
        "Mileage (miles)",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.0f",
    )

    horsepower = st.number_input(
        "Horsepower (HP)",
        min_value=20.0,
        max_value=2000.0,
        value=200.0,
        step=5.0,
    )

    displacement = st.number_input(
        "Engine Displacement (L)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.1,
    )

with right:
    fuel_type = st.selectbox(
        "Fuel Type",
        fuel_options,
        index=0,
    )

    transmission = st.selectbox(
        "Transmission",
        transmission_options,
        index=0,
    )

    v_engine_label_map = {
        "False": "No",
        "True": "Yes",
    }
    v_engine_choice = st.selectbox(
        "V-Type Engine",
        options=v_engine_classes,
        format_func=lambda x: v_engine_label_map.get(str(x), str(x)),
    )

    accident = st.selectbox(
        "Accident / Damage History",
        options=[
            "None reported",
            "At least 1 accident or damage reported",
        ],
    )

    clean_title = st.selectbox(
        "Clean Title",
        options=["Yes", "No"],
    )

st.markdown("")

predict = st.button(
    "Predict Used Car Price",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Prediction
# ============================================================

if predict:
    try:
        X_input = build_features(
            brand=brand,
            model_year=model_year,
            mileage=mileage,
            horsepower=horsepower,
            displacement=displacement,
            is_v_engine=str(v_engine_choice),
            fuel_type=fuel_type,
            transmission=transmission,
            accident=accident,
            clean_title=clean_title,
        )

        prediction = float(model.predict(X_input)[0])
        prediction = max(0.0, prediction)

        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">Estimated Used Car Price</div>
                <div class="price-value">${prediction:,.0f}</div>
                <div class="muted">Predicted by the saved {model_name} model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Vehicle Age", f"{reference_year - int(model_year)} years")
        with c2:
            st.metric(
                "Mileage / Year",
                f"{(mileage / (reference_year - int(model_year)) if model_year != reference_year else mileage):,.0f}",
            )
        with c3:
            st.metric("Horsepower", f"{horsepower:,.0f} HP")

        with st.expander("Model Input Features"):
            st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)

    except Exception as exc:
        st.error("Prediction could not be generated.")
        st.code(str(exc))


# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "Inference uses the saved complete pipeline from "
    "`artifacts/models/best_model.pkl` and the preprocessing metadata generated by the notebook."
)
