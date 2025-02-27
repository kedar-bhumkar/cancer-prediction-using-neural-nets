import torch
import numpy as np
from torch.utils.data import DataLoader
from util import (RiskDataset, risk_mapping, input_dim,  model_name, 
                  device, get_data, get_best_params, 
                 get_criterion, get_optimizer, get_model)
from ann_risk_hp_search import train






# Create final model with best parameters
X_train, y_train, X_val, y_val = get_data()
best_params = get_best_params()
train_dataset = RiskDataset(X_train, y_train)
val_dataset = RiskDataset(X_val, y_val)
train_loader = DataLoader(dataset=train_dataset, batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=best_params['batch_size'], shuffle=False)

model = get_model(best_params['hidden_size'])
criterion = get_criterion()
optimizer = get_optimizer(model, best_params['learning_rate'])
final_model, train_losses, val_losses = train(model, train_loader, val_loader, 
                                            criterion, optimizer, best_params['num_epochs'])


# Example usage for a single new patient
# (Assume we have a new patient's 10-dimensional feature vector from SOAP notes)
new_patient_features = np.random.rand(1, input_dim).astype(np.float32)
new_patient_tensor = torch.tensor(new_patient_features).to(device)

final_model.eval()
with torch.no_grad():
    logits = final_model(new_patient_tensor)
    predicted_class = torch.argmax(logits, dim=1).item()

print("Predicted Risk Level:", risk_mapping[predicted_class])


# Save the final model and parameters
torch.save(final_model.cpu().state_dict(), model_name)
final_model.to(device)  # Move back to GPU if needed
print(f"Model saved as {model_name}")

