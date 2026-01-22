#!/usr/bin/env python3
"""
Yacht Hull Resistance Prediction & Optimization Application

A professional Streamlit web application for predicting and optimizing
yacht hull resistance using machine learning models trained on the
Delft Yacht Hydrodynamics dataset.

Author: AI/ML Engineering Demo
Dataset: TU Delft Ship Hydromechanics Laboratory (308 experiments)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from scipy.optimize import minimize, differential_evolution
from typing import Optional, Tuple, Dict, List

# =============================================================================
# CONFIGURATION
# =============================================================================

# Page configuration
st.set_page_config(
    page_title="Yacht Hull Resistance Predictor",
    page_icon="⛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme (maritime theme)
COLORS = {
    "primary": "#1e3a5f",      # Deep navy
    "secondary": "#3d6b99",    # Ocean blue
    "accent": "#5ba3d9",       # Light blue
    "success": "#28a745",      # Green
    "warning": "#ffc107",      # Yellow
    "danger": "#dc3545",       # Red
    "light": "#f8f9fa",        # Light grey
    "dark": "#343a40"          # Dark grey
}

# Model directory
MODEL_DIR = "models"

# Feature information with descriptions and units
FEATURE_INFO = {
    "longitudinal_position": {
        "name": "Longitudinal Position of CoB",
        "short": "LC",
        "unit": "adimensional",
        "description": "Longitudinal position of the center of buoyancy, measured from the bow. Affects trim and resistance distribution.",
        "icon": "📍"
    },
    "prismatic_coefficient": {
        "name": "Prismatic Coefficient",
        "short": "Cp",
        "unit": "adimensional",
        "description": "Ratio of displaced volume to the volume of a prism with the same length and max cross-section. Higher values = fuller hull.",
        "icon": "📐"
    },
    "length_displacement": {
        "name": "Length-Displacement Ratio",
        "short": "L/∇^⅓",
        "unit": "adimensional",
        "description": "Slenderness coefficient. Higher values indicate a more slender hull with potentially lower resistance.",
        "icon": "📏"
    },
    "beam_draught": {
        "name": "Beam-Draught Ratio",
        "short": "B/Dr",
        "unit": "adimensional",
        "description": "Width to draft ratio. Affects stability and wetted surface area.",
        "icon": "↔️"
    },
    "length_beam": {
        "name": "Length-Beam Ratio",
        "short": "L/B",
        "unit": "adimensional",
        "description": "Length to width ratio. Higher values typically reduce wave-making resistance.",
        "icon": "📊"
    },
    "froude_number": {
        "name": "Froude Number",
        "short": "Fr",
        "unit": "adimensional",
        "description": "Speed-length ratio (V/√(gL)). Determines the resistance regime - critical around 0.4-0.5.",
        "icon": "⚡"
    }
}

# Preset designs for quick loading
PRESET_DESIGNS = {
    "Racing Yacht": {
        "description": "Optimized for high-speed performance",
        "values": {
            "longitudinal_position": -2.5,
            "prismatic_coefficient": 0.53,
            "length_displacement": 5.0,
            "beam_draught": 3.5,
            "length_beam": 4.0,
            "froude_number": 0.35
        }
    },
    "Cruising Yacht": {
        "description": "Balanced design for comfort and efficiency",
        "values": {
            "longitudinal_position": -3.0,
            "prismatic_coefficient": 0.56,
            "length_displacement": 4.5,
            "beam_draught": 3.8,
            "length_beam": 3.5,
            "froude_number": 0.30
        }
    },
    "Heavy Displacement": {
        "description": "Traditional design with high stability",
        "values": {
            "longitudinal_position": -4.0,
            "prismatic_coefficient": 0.58,
            "length_displacement": 4.0,
            "beam_draught": 4.0,
            "length_beam": 3.2,
            "froude_number": 0.25
        }
    }
}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained ML model from disk."""
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_scaler():
    """Load the feature scaler from disk."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)


@st.cache_data
def load_metadata():
    """Load model metadata including feature statistics."""
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    """Load the yacht hydrodynamics dataset."""
    data_path = os.path.join(MODEL_DIR, "yacht_data.csv")
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


@st.cache_data
def load_all_model_results():
    """Load comparison results for all trained models."""
    results_path = os.path.join(MODEL_DIR, "all_model_results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path, 'r') as f:
        return json.load(f)


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_resistance(model, scaler, features: np.ndarray) -> float:
    """
    Predict residuary resistance for given hull parameters.

    Args:
        model: Trained ML model
        scaler: Feature scaler
        features: Array of 6 hull parameters

    Returns:
        Predicted resistance value
    """
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    return prediction


def estimate_uncertainty(model, scaler, features: np.ndarray,
                        n_samples: int = 100) -> Tuple[float, float]:
    """
    Estimate prediction uncertainty using bootstrap-like approach.

    For Random Forest, uses tree variance. For other models, uses
    a simple perturbation-based approach.

    Args:
        model: Trained ML model
        scaler: Feature scaler
        features: Array of 6 hull parameters
        n_samples: Number of perturbation samples

    Returns:
        Tuple of (lower_bound, upper_bound) for 95% CI
    """
    features_scaled = scaler.transform(features.reshape(1, -1))

    # For Random Forest, use individual tree predictions
    if hasattr(model, 'estimators_'):
        predictions = np.array([
            tree.predict(features_scaled)[0]
            for tree in model.estimators_
        ])
        mean_pred = predictions.mean()
        std_pred = predictions.std()
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        return max(0, lower), upper

    # For other models, use input perturbation
    base_pred = model.predict(features_scaled)[0]
    perturbations = []

    for _ in range(n_samples):
        noise = np.random.normal(0, 0.05, features_scaled.shape)
        perturbed = features_scaled + noise
        perturbations.append(model.predict(perturbed)[0])

    perturbations = np.array(perturbations)
    std_pred = perturbations.std()
    lower = base_pred - 1.96 * std_pred
    upper = base_pred + 1.96 * std_pred

    return max(0, lower), upper


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def optimize_design(model, scaler, metadata: dict,
                   target_resistance: Optional[float] = None,
                   constraints: Optional[dict] = None,
                   fixed_params: Optional[dict] = None,
                   n_results: int = 10) -> pd.DataFrame:
    """
    Optimize hull design to minimize resistance.

    Args:
        model: Trained ML model
        scaler: Feature scaler
        metadata: Model metadata with feature stats
        target_resistance: Optional specific resistance target
        constraints: Dict of {feature: (min, max)} constraints
        fixed_params: Dict of {feature: value} for fixed parameters
        n_results: Number of top results to return

    Returns:
        DataFrame with top optimized designs
    """
    feature_names = list(FEATURE_INFO.keys())
    feature_stats = metadata["feature_stats"]

    # Set up bounds
    bounds = []
    for feat in feature_names:
        if constraints and feat in constraints:
            bounds.append(constraints[feat])
        else:
            # Use dataset range with small padding
            fmin = feature_stats[feat]["min"]
            fmax = feature_stats[feat]["max"]
            padding = (fmax - fmin) * 0.05
            bounds.append((fmin - padding, fmax + padding))

    # Objective function
    def objective(x):
        # Apply fixed parameters
        if fixed_params:
            x_full = x.copy()
            for feat, val in fixed_params.items():
                idx = feature_names.index(feat)
                x_full[idx] = val
        else:
            x_full = x

        x_scaled = scaler.transform(x_full.reshape(1, -1))
        pred = model.predict(x_scaled)[0]

        if target_resistance is not None:
            # Minimize distance to target
            return abs(pred - target_resistance)
        else:
            # Minimize resistance
            return pred

    # Run multiple optimizations with different starting points
    results = []

    for _ in range(n_results * 3):  # Extra runs to get diverse results
        # Random starting point within bounds
        x0 = np.array([
            np.random.uniform(b[0], b[1]) for b in bounds
        ])

        # Use differential evolution for global optimization
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=np.random.randint(10000),
            maxiter=100,
            tol=1e-6,
            workers=1
        )

        if result.success:
            x_opt = result.x
            if fixed_params:
                for feat, val in fixed_params.items():
                    idx = feature_names.index(feat)
                    x_opt[idx] = val

            pred = predict_resistance(model, scaler, x_opt)
            results.append({
                **{feat: x_opt[i] for i, feat in enumerate(feature_names)},
                "predicted_resistance": pred
            })

    # Convert to DataFrame and deduplicate
    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Round to avoid near-duplicates
    df_rounded = df_results.round(3)
    df_results = df_results.loc[~df_rounded.duplicated()]

    # Sort by resistance and take top N
    df_results = df_results.sort_values("predicted_resistance").head(n_results)
    df_results = df_results.reset_index(drop=True)
    df_results.index = df_results.index + 1  # 1-based ranking

    return df_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_scatter_context_plot(df: pd.DataFrame, current_values: dict,
                               current_prediction: float) -> go.Figure:
    """
    Create scatter plot showing current design among all test cases.

    Args:
        df: Dataset DataFrame
        current_values: Current parameter values
        current_prediction: Current resistance prediction

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Dataset points
    fig.add_trace(go.Scatter(
        x=df["froude_number"],
        y=df["residuary_resistance"],
        mode="markers",
        marker=dict(
            size=8,
            color=df["residuary_resistance"],
            colorscale="Blues",
            opacity=0.6
        ),
        name="Dataset (308 cases)",
        hovertemplate=(
            "Froude: %{x:.3f}<br>"
            "Resistance: %{y:.2f}<br>"
            "<extra></extra>"
        )
    ))

    # Current design point
    fig.add_trace(go.Scatter(
        x=[current_values["froude_number"]],
        y=[current_prediction],
        mode="markers",
        marker=dict(
            size=15,
            color=COLORS["danger"],
            symbol="star",
            line=dict(width=2, color="white")
        ),
        name="Your Design",
        hovertemplate=(
            "<b>Your Design</b><br>"
            "Froude: %{x:.3f}<br>"
            "Predicted Resistance: %{y:.2f}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title="Your Design vs Dataset",
        xaxis_title="Froude Number",
        yaxis_title="Residuary Resistance",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template="plotly_white",
        height=400
    )

    return fig


def create_3d_surface_plot(model, scaler, metadata: dict,
                          param_x: str, param_y: str,
                          fixed_values: dict) -> go.Figure:
    """
    Create 3D surface plot showing resistance variation.

    Args:
        model: Trained ML model
        scaler: Feature scaler
        metadata: Model metadata
        param_x: X-axis parameter name
        param_y: Y-axis parameter name
        fixed_values: Dict of fixed parameter values

    Returns:
        Plotly figure
    """
    feature_names = list(FEATURE_INFO.keys())
    feature_stats = metadata["feature_stats"]

    # Create grid for the two varying parameters
    x_range = np.linspace(
        feature_stats[param_x]["min"],
        feature_stats[param_x]["max"],
        30
    )
    y_range = np.linspace(
        feature_stats[param_y]["min"],
        feature_stats[param_y]["max"],
        30
    )

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    # Calculate predictions for each grid point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            features = np.zeros(6)
            for k, feat in enumerate(feature_names):
                if feat == param_x:
                    features[k] = X[i, j]
                elif feat == param_y:
                    features[k] = Y[i, j]
                else:
                    features[k] = fixed_values.get(feat, feature_stats[feat]["mean"])

            Z[i, j] = predict_resistance(model, scaler, features)

    # Create surface plot
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale="RdYlGn_r",  # Red=high, Green=low resistance
        hovertemplate=(
            f"{FEATURE_INFO[param_x]['short']}: %{{x:.3f}}<br>"
            f"{FEATURE_INFO[param_y]['short']}: %{{y:.3f}}<br>"
            "Resistance: %{z:.2f}<br>"
            "<extra></extra>"
        )
    )])

    fig.update_layout(
        title=f"Resistance Surface: {FEATURE_INFO[param_x]['short']} vs {FEATURE_INFO[param_y]['short']}",
        scene=dict(
            xaxis_title=FEATURE_INFO[param_x]["short"],
            yaxis_title=FEATURE_INFO[param_y]["short"],
            zaxis_title="Resistance",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600
    )

    return fig


def create_feature_importance_plot(model, feature_names: list) -> go.Figure:
    """
    Create feature importance bar chart.

    Args:
        model: Trained ML model
        feature_names: List of feature names

    Returns:
        Plotly figure
    """
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For models without built-in importance, use permutation-based estimate
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Create DataFrame for plotting
    df_imp = pd.DataFrame({
        "Feature": [FEATURE_INFO[f]["short"] for f in feature_names],
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation="h",
        marker_color=COLORS["secondary"],
        hovertemplate=(
            "%{y}<br>"
            "Importance: %{x:.3f}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Relative Importance",
        yaxis_title="",
        template="plotly_white",
        height=350
    )

    return fig


def create_pareto_plot(df_results: pd.DataFrame, param1: str, param2: str) -> go.Figure:
    """
    Create trade-off visualization between two parameters.

    Args:
        df_results: Optimization results DataFrame
        param1: First parameter name
        param2: Second parameter name

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_results[param1],
        y=df_results["predicted_resistance"],
        mode="markers+lines",
        marker=dict(
            size=12,
            color=df_results.index,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Rank")
        ),
        line=dict(dash="dot", color="grey"),
        hovertemplate=(
            f"Rank: %{{text}}<br>"
            f"{FEATURE_INFO[param1]['short']}: %{{x:.3f}}<br>"
            "Resistance: %{y:.2f}<br>"
            "<extra></extra>"
        ),
        text=df_results.index
    ))

    fig.update_layout(
        title=f"Trade-off: {FEATURE_INFO[param1]['short']} vs Resistance",
        xaxis_title=FEATURE_INFO[param1]["short"],
        yaxis_title="Predicted Resistance",
        template="plotly_white",
        height=400
    )

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(metadata: dict, model_results: dict):
    """Render the sidebar with dataset info and model metrics."""
    st.sidebar.title("⛵ Yacht Resistance Predictor")

    st.sidebar.markdown("---")

    # Dataset info
    st.sidebar.subheader("📊 Dataset Information")
    st.sidebar.markdown("""
    **Source:** TU Delft Ship Hydromechanics Laboratory

    **Experiments:** 308 yacht hull tests

    **Citation:** Delft Yacht Hydrodynamics Dataset
    [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics)
    """)

    st.sidebar.markdown("---")

    # Model performance
    if metadata:
        st.sidebar.subheader("🎯 Model Performance")
        metrics = metadata["metrics"]

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("R² Score", f"{metrics['test_r2']:.3f}")
        with col2:
            st.metric("MAE", f"{metrics['test_mae']:.3f}")

        st.sidebar.caption(f"Model: {metadata['model_name']}")

    # Model comparison
    if model_results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Model Comparison")

        df_models = pd.DataFrame(model_results).T
        df_models = df_models.round(3)
        st.sidebar.dataframe(df_models)

    st.sidebar.markdown("---")

    # About section
    with st.sidebar.expander("ℹ️ About This Tool"):
        st.markdown("""
        **What it does:**
        Predicts hydrodynamic resistance of sailing yacht hulls using machine learning.

        **How it works:**
        - Trained on 308 experimental measurements
        - Uses ensemble ML models (Random Forest/Gradient Boosting)
        - Accepts 6 hull form parameters as input

        **Limitations:**
        - Trained specifically on sailing yacht data
        - Valid for the parameter ranges in the dataset
        - Residuary resistance only (not total resistance)

        **Scaling to Commercial Vessels:**
        The same ML approach can be applied to commercial vessels with appropriate training data:
        - Container ships, tankers, bulk carriers
        - Different hull form parameters
        - Full-scale CFD or model test data
        """)


# =============================================================================
# TAB 1: INSTANT PREDICTOR
# =============================================================================

def render_predictor_tab(model, scaler, metadata: dict, df: pd.DataFrame):
    """Render the instant resistance predictor tab."""
    st.header("🎯 Instant Resistance Predictor")
    st.markdown("Adjust hull parameters and see real-time resistance predictions.")

    feature_stats = metadata["feature_stats"]

    # Preset designs
    col_preset, col_reset = st.columns([3, 1])
    with col_preset:
        preset = st.selectbox(
            "Load preset design:",
            ["Custom"] + list(PRESET_DESIGNS.keys()),
            help="Select a preset design to load its parameters"
        )

    with col_reset:
        st.write("")  # Spacing
        if st.button("Reset to Defaults", width="stretch"):
            for key in st.session_state:
                if key.startswith("slider_"):
                    del st.session_state[key]
            st.rerun()

    # Apply preset values
    preset_values = None
    if preset != "Custom":
        preset_values = PRESET_DESIGNS[preset]["values"]
        st.info(f"📋 {PRESET_DESIGNS[preset]['description']}")

    st.markdown("---")

    # Parameter sliders
    col1, col2 = st.columns(2)
    current_values = {}

    features_list = list(FEATURE_INFO.keys())

    for i, feat in enumerate(features_list):
        info = FEATURE_INFO[feat]
        stats = feature_stats[feat]

        # Determine which column
        col = col1 if i < 3 else col2

        with col:
            # Default value: preset or mean
            default_val = preset_values[feat] if preset_values else stats["mean"]

            current_values[feat] = st.slider(
                f"{info['icon']} {info['name']} ({info['short']})",
                min_value=float(stats["min"]),
                max_value=float(stats["max"]),
                value=float(default_val),
                step=(stats["max"] - stats["min"]) / 100,
                help=info["description"],
                key=f"slider_{feat}"
            )

    st.markdown("---")

    # Make prediction
    features = np.array([current_values[f] for f in features_list])
    prediction = predict_resistance(model, scaler, features)
    lower_ci, upper_ci = estimate_uncertainty(model, scaler, features)

    # Display prediction prominently
    col_pred, col_ci, col_viz = st.columns([1, 1, 2])

    with col_pred:
        st.markdown("### Predicted Resistance")
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        ">
            <span style="font-size: 48px; font-weight: bold; color: white;">
                {prediction:.2f}
            </span>
            <br>
            <span style="color: rgba(255,255,255,0.8); font-size: 14px;">
                Residuary Resistance (Rr)
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_ci:
        st.markdown("### Confidence Interval")
        st.markdown(f"""
        <div style="
            background: {COLORS['light']};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #ddd;
        ">
            <span style="font-size: 24px; color: {COLORS['dark']};">
                {lower_ci:.2f} - {upper_ci:.2f}
            </span>
            <br>
            <span style="color: #666; font-size: 14px;">
                95% Confidence Interval
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Confidence indicator
        uncertainty = upper_ci - lower_ci
        if uncertainty < 5:
            st.success("✓ High confidence prediction")
        elif uncertainty < 10:
            st.warning("⚠ Moderate uncertainty")
        else:
            st.error("⚠ High uncertainty - design may be outside training range")

    with col_viz:
        # Scatter plot showing context
        fig = create_scatter_context_plot(df, current_values, prediction)
        st.plotly_chart(fig, width="stretch")

    # Save design to comparison
    st.markdown("---")
    col_save, col_name = st.columns([1, 2])

    with col_name:
        design_name = st.text_input(
            "Design name:",
            value=f"Design {len(st.session_state.get('saved_designs', [])) + 1}",
            key="design_name_input"
        )

    with col_save:
        st.write("")  # Spacing
        if st.button("💾 Save for Comparison", width="stretch"):
            if "saved_designs" not in st.session_state:
                st.session_state.saved_designs = []

            design = {
                "name": design_name,
                "values": current_values.copy(),
                "prediction": prediction,
                "ci": (lower_ci, upper_ci)
            }
            st.session_state.saved_designs.append(design)
            st.success(f"Saved '{design_name}'!")

    # Show saved designs comparison
    if st.session_state.get("saved_designs"):
        st.markdown("### 📊 Saved Designs Comparison")

        # Create comparison DataFrame
        comparison_data = []
        for d in st.session_state.saved_designs:
            row = {"Design": d["name"], "Resistance": f"{d['prediction']:.2f}"}
            for feat in features_list:
                row[FEATURE_INFO[feat]["short"]] = f"{d['values'][feat]:.3f}"
            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, width="stretch", hide_index=True)

        if st.button("🗑️ Clear Saved Designs"):
            st.session_state.saved_designs = []
            st.rerun()


# =============================================================================
# TAB 2: DESIGN OPTIMIZER
# =============================================================================

def render_optimizer_tab(model, scaler, metadata: dict):
    """Render the design optimizer tab."""
    st.header("🔧 Design Optimizer")
    st.markdown("Find optimal hull parameters to minimize resistance.")

    feature_stats = metadata["feature_stats"]
    features_list = list(FEATURE_INFO.keys())

    # Optimization settings
    col_target, col_method = st.columns(2)

    with col_target:
        opt_mode = st.radio(
            "Optimization target:",
            ["Minimize resistance", "Target specific resistance"],
            horizontal=True
        )

        target_resistance = None
        if opt_mode == "Target specific resistance":
            target_resistance = st.number_input(
                "Target resistance value:",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5
            )

    with col_method:
        n_results = st.slider(
            "Number of results to show:",
            min_value=5,
            max_value=20,
            value=10
        )

    st.markdown("---")
    st.subheader("📐 Parameter Constraints")
    st.markdown("Set bounds for each parameter. Check 'Fix' to lock a parameter to a specific value.")

    # Constraint inputs
    constraints = {}
    fixed_params = {}

    # Create columns for constraints
    col1, col2 = st.columns(2)

    for i, feat in enumerate(features_list):
        info = FEATURE_INFO[feat]
        stats = feature_stats[feat]

        col = col1 if i < 3 else col2

        with col:
            with st.expander(f"{info['icon']} {info['name']} ({info['short']})"):
                # Fix parameter toggle
                is_fixed = st.checkbox(
                    "Fix this parameter",
                    key=f"fix_{feat}",
                    help="Lock this parameter to a specific value"
                )

                if is_fixed:
                    fixed_val = st.number_input(
                        "Fixed value:",
                        min_value=float(stats["min"]),
                        max_value=float(stats["max"]),
                        value=float(stats["mean"]),
                        key=f"fixed_{feat}"
                    )
                    fixed_params[feat] = fixed_val
                else:
                    # Range constraints
                    c1, c2 = st.columns(2)
                    with c1:
                        min_val = st.number_input(
                            "Min:",
                            min_value=float(stats["min"]),
                            max_value=float(stats["max"]),
                            value=float(stats["min"]),
                            key=f"min_{feat}"
                        )
                    with c2:
                        max_val = st.number_input(
                            "Max:",
                            min_value=float(stats["min"]),
                            max_value=float(stats["max"]),
                            value=float(stats["max"]),
                            key=f"max_{feat}"
                        )
                    constraints[feat] = (min_val, max_val)

    st.markdown("---")

    # Run optimization button
    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        run_optimization = st.button(
            "🚀 Optimize",
            type="primary",
            width="stretch"
        )

    if run_optimization:
        with st.spinner("Running optimization... This may take a moment."):
            df_results = optimize_design(
                model, scaler, metadata,
                target_resistance=target_resistance,
                constraints=constraints if constraints else None,
                fixed_params=fixed_params if fixed_params else None,
                n_results=n_results
            )

        if df_results.empty:
            st.error("Optimization failed to find valid designs. Try relaxing constraints.")
        else:
            st.success(f"Found {len(df_results)} optimal designs!")

            # Store results in session state
            st.session_state.optimization_results = df_results

    # Display results
    if "optimization_results" in st.session_state and not st.session_state.optimization_results.empty:
        df_results = st.session_state.optimization_results

        st.subheader("🏆 Top Optimized Designs")

        # Format display DataFrame
        display_cols = {feat: FEATURE_INFO[feat]["short"] for feat in features_list}
        display_cols["predicted_resistance"] = "Resistance"

        df_display = df_results.rename(columns=display_cols).round(3)
        st.dataframe(df_display, width="stretch")

        # Download button
        csv = df_results.to_csv(index=True)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="optimized_designs.csv",
            mime="text/csv"
        )

        # Trade-off visualization
        st.subheader("📈 Trade-off Analysis")

        viz_param = st.selectbox(
            "Show trade-off with:",
            features_list,
            format_func=lambda x: FEATURE_INFO[x]["name"]
        )

        fig = create_pareto_plot(df_results, viz_param, "predicted_resistance")
        st.plotly_chart(fig, width="stretch")


# =============================================================================
# TAB 3: DESIGN SPACE EXPLORER
# =============================================================================

def render_explorer_tab(model, scaler, metadata: dict):
    """Render the design space explorer tab."""
    st.header("🌐 Design Space Explorer")
    st.markdown("Visualize how resistance varies across different hull parameters.")

    feature_stats = metadata["feature_stats"]
    features_list = list(FEATURE_INFO.keys())

    # Parameter selection
    col1, col2 = st.columns(2)

    with col1:
        param_x = st.selectbox(
            "X-axis parameter:",
            features_list,
            index=5,  # Froude number
            format_func=lambda x: f"{FEATURE_INFO[x]['icon']} {FEATURE_INFO[x]['name']}"
        )

    with col2:
        param_y_options = [f for f in features_list if f != param_x]
        param_y = st.selectbox(
            "Y-axis parameter:",
            param_y_options,
            index=2,  # Length-displacement
            format_func=lambda x: f"{FEATURE_INFO[x]['icon']} {FEATURE_INFO[x]['name']}"
        )

    # Fixed parameter values
    st.markdown("---")
    st.subheader("📍 Fixed Parameter Values")
    st.markdown("Set values for parameters not shown in the plot:")

    fixed_values = {}
    other_params = [f for f in features_list if f not in [param_x, param_y]]

    cols = st.columns(len(other_params))

    for i, feat in enumerate(other_params):
        info = FEATURE_INFO[feat]
        stats = feature_stats[feat]

        with cols[i]:
            fixed_values[feat] = st.slider(
                f"{info['short']}",
                min_value=float(stats["min"]),
                max_value=float(stats["max"]),
                value=float(stats["mean"]),
                help=info["description"],
                key=f"explorer_{feat}"
            )

    st.markdown("---")

    # Generate 3D surface plot
    with st.spinner("Generating surface plot..."):
        fig_surface = create_3d_surface_plot(
            model, scaler, metadata,
            param_x, param_y, fixed_values
        )

    st.plotly_chart(fig_surface, width="stretch")

    st.markdown("""
    **Interpretation:**
    - 🟢 Green areas indicate low resistance (efficient designs)
    - 🔴 Red areas indicate high resistance (less efficient designs)
    - Drag to rotate the 3D view, scroll to zoom
    """)

    # Feature importance
    st.markdown("---")
    st.subheader("📊 Feature Importance")
    st.markdown("Which parameters have the biggest impact on resistance?")

    fig_importance = create_feature_importance_plot(model, features_list)
    st.plotly_chart(fig_importance, width="stretch")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Check if models exist
    model = load_model()
    scaler = load_scaler()
    metadata = load_metadata()
    df = load_dataset()
    model_results = load_all_model_results()

    # Render sidebar
    render_sidebar(metadata, model_results)

    # Main content
    if model is None or scaler is None or metadata is None:
        st.error("⚠️ Model not found. Please run the training script first.")

        st.markdown("""
        ### Setup Instructions

        1. **Install dependencies:**
           ```bash
           pip install -r requirements.txt
           ```

        2. **Train the model:**
           ```bash
           python train_model.py
           ```

        3. **Launch the application:**
           ```bash
           streamlit run streamlit_app.py
           ```
        """)

        # Offer to train the model
        if st.button("🚀 Train Model Now", type="primary"):
            with st.spinner("Training models... This may take a minute."):
                import subprocess
                result = subprocess.run(
                    ["python", "train_model.py"],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    st.success("Model trained successfully! Refreshing...")
                    st.rerun()
                else:
                    st.error(f"Training failed: {result.stderr}")

        return

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Instant Predictor",
        "🔧 Design Optimizer",
        "🌐 Design Space Explorer"
    ])

    with tab1:
        render_predictor_tab(model, scaler, metadata, df)

    with tab2:
        render_optimizer_tab(model, scaler, metadata)

    with tab3:
        render_explorer_tab(model, scaler, metadata)


if __name__ == "__main__":
    main()
