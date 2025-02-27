import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import json
import torch.nn as nn
import torch.optim as optim
from model import RiskModel
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


# Model parameters
input_dim = 23
hidden_dim = 32
output_dim = 3
model_name = "risk_assessment_model.pth"
best_params_name = "best_parameters.json"
use_random_data = False
dataset_path = "cancer_patient_data_sets.csv"

risk_mapping = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a custom Dataset
class RiskDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.LongTensor(y).to(device)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


def get_data():
    if(use_random_data):
        # For reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # ---------------------
        # GENERATE SYNTHETIC DATA
        # ---------------------
        # Let's assume each patient has 10 features derived from their SOAP notes
        # (e.g., vital signs, symptom scores, lab results, etc.)
        num_samples = 1000
        # Synthetic features
        X = np.random.rand(num_samples, input_dim).astype(np.float32)
        # Synthetic labels (3 possible classes)
        y = np.random.randint(0, output_dim, size=num_samples)

        # Convert to Torch tensors
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y, dtype=torch.long)  # Specify dtype as long

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )

        print("Training samples:", X_train.shape[0])
        print("Validation samples:", X_val.shape[0])
    else:
        # Load the data from the CSV file
        X_train, y_train, X_val, y_val = load_data()
    
    return X_train, y_train, X_val, y_val

def get_best_params():
     # Load the best parameters
    with open(best_params_name, 'r') as f:
        best_params = json.load(f)
    return best_params

def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)

def get_model(hidden_size):
    model = RiskModel(input_dim, hidden_size, output_dim).to(device)
    return model

def load_data():
    # Read the CSV file
    
    # Read the dataset
    df = pd.read_csv(dataset_path)
    
    # Drop index and Patient Id columns
    df = df.drop(['index', 'Patient Id'], axis=1)
    
    # Create level mapping
    level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    
    # Convert Level to numeric using the mapping
    df['Level'] = df['Level'].map(level_mapping)
    
    # Separate features (X) and target (y)
    X = df.drop('Level', axis=1)
    y = df['Level']
    

    
    # Convert to numpy arrays
    X = X.values.astype(np.float32)
    y = y.values.astype(np.int64)
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors with explicit dtypes
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    print("Training samples:", X_train.shape[0])
    print("Validation samples:", X_val.shape[0])
    
    return X_train, y_train, X_val, y_val
