import torch
import numpy as np
from model import RiskModel
from util import input_dim, output_dim, risk_mapping, model_name,  device, get_best_params, dataset_path
import pandas as pd
import random


def load_model_and_predict(features=None):
    if features is None:
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Drop index and Patient Id columns
        df = df.drop(['index', 'Patient Id', 'Level'], axis=1)
        
        # Randomly select one patient
        random_index = random.randint(0, len(df) - 1)
        features = df.iloc[random_index:random_index+1].values.astype(np.float32)
        
        print(f"Selected patient at index: {random_index}")
    
    best_params = get_best_params()
    hidden_dim = best_params['hidden_size']

    # Initialize the model
    model = RiskModel(input_dim, hidden_dim, output_dim).to(device)
    
    # Load the saved model state
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()

    # Convert input features to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    # Make prediction
    with torch.no_grad():
        logits = model(features_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        
    return predicted_class, risk_mapping[predicted_class]

if __name__ == "__main__":
  
    predicted_class, risk_level = load_model_and_predict(None)
    
   
    print(f"Predicted class: {predicted_class}")
    print(f"Risk level: {risk_level}")
