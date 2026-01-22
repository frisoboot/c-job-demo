# Yacht Hull Resistance Predictor

A professional Streamlit web application for predicting and optimizing yacht hull resistance using machine learning. Built on the Delft Yacht Hydrodynamics dataset from TU Delft Ship Hydromechanics Laboratory.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This application demonstrates how AI/ML can be applied to naval architecture and hull design optimization. It allows engineers to:

- **Predict** hydrodynamic resistance for custom hull configurations
- **Optimize** hull parameters to minimize resistance
- **Explore** the design space through interactive 3D visualizations

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone or download this repository
cd c-job-demo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Step 1: Train the ML models (downloads dataset and trains models)
python train_model.py

# Step 2: Launch the Streamlit app
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Features

### Tab 1: Instant Resistance Predictor
- Interactive sliders for 6 hull parameters
- Real-time resistance predictions as you adjust values
- 95% confidence intervals for predictions
- Visual comparison with 308 experimental data points
- Preset designs (Racing, Cruising, Heavy Displacement)
- Save and compare multiple designs

### Tab 2: Design Optimizer
- Automatic optimization using differential evolution
- Set constraints on parameter ranges
- Lock specific parameters to fixed values
- Top 10 optimal designs ranked by resistance
- Export results to CSV
- Trade-off analysis visualizations

### Tab 3: Design Space Explorer
- Interactive 3D surface plots
- Visualize how resistance varies with any 2 parameters
- Adjust fixed parameters in real-time
- Feature importance analysis
- Color-coded resistance (green=low, red=high)

## Technical Approach

### Dataset
- **Source**: Delft Yacht Hydrodynamics Dataset (UCI ML Repository)
- **Size**: 308 experiments from TU Delft Ship Hydromechanics Laboratory
- **Features**: 6 hull form parameters
- **Target**: Residuary resistance per unit weight of displacement

### Input Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Longitudinal Position of CoB | LC | Position of center of buoyancy |
| Prismatic Coefficient | Cp | Hull fullness ratio |
| Length-Displacement Ratio | L/∇^⅓ | Slenderness coefficient |
| Beam-Draught Ratio | B/Dr | Width to draft ratio |
| Length-Beam Ratio | L/B | Length to width ratio |
| Froude Number | Fr | Speed-length ratio |

### Machine Learning Pipeline

1. **Data Processing**
   - Feature standardization using StandardScaler
   - 80/20 train/test split
   - Cross-validation for model selection

2. **Models Trained**
   - Random Forest Regressor (200 trees)
   - Gradient Boosting Regressor
   - Neural Network (MLP with 3 hidden layers)

3. **Model Selection**
   - Best model selected based on test R² score
   - Typical performance: R² > 0.99, MAE < 1.0

4. **Uncertainty Estimation**
   - For Random Forest: tree variance
   - 95% confidence intervals provided

### Optimization

- Uses SciPy's `differential_evolution` for global optimization
- Multiple random starts for diverse solutions
- Supports parameter constraints and fixed values

## Demo Presentation Guide

### For Maritime Engineers

**Key Value Proposition:**
"This tool demonstrates how machine learning can accelerate hull design iteration from days to seconds, while maintaining the accuracy of traditional CFD or tank testing."

**Talking Points:**
1. **Speed**: Instant predictions vs hours/days for CFD
2. **Exploration**: Quickly explore thousands of design variations
3. **Optimization**: Find optimal designs automatically
4. **Validation**: Compare predictions against 308 real experiments

**Live Demo Flow:**
1. Start with a preset design (Racing Yacht)
2. Show how changing Froude number affects resistance
3. Run the optimizer with constraints
4. Explore the 3D design space
5. Highlight the feature importance chart

### Anticipated Questions & Answers

**Q: How accurate are these predictions?**
A: The model achieves R² > 0.99 on test data, meaning it explains over 99% of variance in resistance. Predictions include 95% confidence intervals.

**Q: Can this replace tank testing or CFD?**
A: It complements rather than replaces traditional methods. Use it for early-stage exploration and down-selection, then validate final designs with CFD or testing.

**Q: What about different vessel types?**
A: This model is trained on sailing yacht data. The same ML approach applies to commercial vessels with appropriate training data (discussed below).

**Q: How does it handle uncertainty?**
A: For Random Forest models, we use tree variance to estimate uncertainty. Predictions outside the training data range show higher uncertainty.

### Scaling to Commercial Vessels

The approach demonstrated here can be extended to commercial maritime vessels:

**Data Requirements:**
- CFD simulation results for various hull forms
- Model test data from towing tanks
- Full-scale sea trial measurements
- Operational data from fleet vessels

**Extended Parameters for Commercial Ships:**
- Block coefficient (Cb)
- Midship section coefficient (Cm)
- Waterplane area coefficient (Cwp)
- Bulbous bow parameters
- Stern shape parameters
- Hull roughness and fouling factors

**Potential Applications:**
- Container ship hull optimization
- Tanker and bulk carrier efficiency
- Cruise ship design optimization
- Ferry and RoRo vessel design
- Offshore support vessel optimization

**Business Value:**
- 1-3% fuel savings = millions in annual savings for large fleets
- Faster design iteration = reduced time-to-market
- Design space exploration = innovation opportunities
- Predictive maintenance = optimized hull cleaning schedules

## File Structure

```
c-job-demo/
├── streamlit_app.py      # Main Streamlit application
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── yacht_hydrodynamics.data  # Downloaded dataset
└── models/              # Trained model artifacts
    ├── best_model.joblib
    ├── scaler.joblib
    ├── model_metadata.json
    ├── yacht_data.csv
    └── all_model_results.json
```

## Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- plotly >= 5.18.0
- requests >= 2.31.0
- joblib >= 1.3.0

## Dataset Citation

Delft Yacht Hydrodynamics Data Set
- Source: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics
- Donor: Ship Hydromechanics Laboratory, TU Delft

## License

MIT License - Feel free to use and modify for your own projects.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

**Built for demonstrating AI/ML capabilities in maritime engineering.**
