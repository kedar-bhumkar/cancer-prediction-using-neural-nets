
import gradio as gr
import numpy as np
import pandas as pd
import sys
import os

# Add the code directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ann_risk_inference import load_model_and_predict

# Read the dataset to understand the feature ranges
df = pd.read_csv('cancer_patient_data_sets.csv')
# Drop index and Patient Id columns
df = df.drop(['index', 'Patient Id', 'Level'], axis=1)

# Get feature names and their ranges
FEATURE_NAMES = df.columns.tolist()
FEATURE_RANGES = {
    col: (df[col].min(), df[col].max())
    for col in FEATURE_NAMES
}

def predict_risk(*feature_values):
    # Convert inputs to numpy array and reshape for model
    features = np.array(feature_values, dtype=np.float32).reshape(1, -1)
    
    # Get prediction
    predicted_class, risk_level = load_model_and_predict(features)
    
    # Format output
    return f"Predicted Risk Level: {risk_level} (Class {predicted_class})"

# Create the interface
inputs = [
    gr.Slider(
        minimum=float(FEATURE_RANGES[name][0]),
        maximum=float(FEATURE_RANGES[name][1]),
        step=1 if name in ['Age'] else 0.1,  # Integer steps for Age
        label=name,
        value=float(df[name].mean())  # Set default value to mean
    )
    for name in FEATURE_NAMES
]

# Create example cases (using 3 random samples from dataset)
example_indices = np.random.choice(len(df), 3, replace=False)
examples = df.iloc[example_indices].values.tolist()

demo = gr.Interface(
    fn=predict_risk,
    inputs=inputs,
    outputs="text",
    title="Cancer Risk Prediction Model",
    description="Enter patient characteristics to predict cancer risk level. Adjust the sliders based on patient data.",
    theme="default",
    examples=examples
)

if __name__ == "__main__":



    # Add the code directory to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ann_risk_inference import load_model_and_predict

    # Read the dataset to understand the feature ranges
    df = pd.read_csv('cancer_patient_data_sets.csv')
    # Drop index and Patient Id columns
    df = df.drop(['index', 'Patient Id', 'Level'], axis=1)

    # Get feature names and their ranges
    FEATURE_NAMES = df.columns.tolist()
    FEATURE_RANGES = {
        col: (df[col].min(), df[col].max())
        for col in FEATURE_NAMES
    }

def predict_risk(*feature_values):
    # Convert inputs to numpy array and reshape for model
    features = np.array(feature_values, dtype=np.float32).reshape(1, -1)
    
    # Get prediction
    predicted_class, risk_level = load_model_and_predict(features)
    
    # Format output
    return f"Predicted Risk Level: {risk_level} (Class {predicted_class})"

# Create the interface
inputs = [
    gr.Slider(
        minimum=float(FEATURE_RANGES[name][0]),
        maximum=float(FEATURE_RANGES[name][1]),
        step=1 if name in ['Age'] else 0.1,  # Integer steps for Age
        label=name,
        value=float(df[name].mean())  # Set default value to mean
    )
    for name in FEATURE_NAMES
]

# Create example cases (using 3 random samples from dataset)
example_indices = np.random.choice(len(df), 3, replace=False)
examples = df.iloc[example_indices].values.tolist()

demo = gr.Interface(
    fn=predict_risk,
    inputs=inputs,
    outputs="text",
    title="Cancer Risk Prediction Model",
    description="Enter patient characteristics to predict cancer risk level. Adjust the sliders based on patient data.",
    theme="default",
    examples=examples
)

if __name__ == "__main__":
    demo.launch() 