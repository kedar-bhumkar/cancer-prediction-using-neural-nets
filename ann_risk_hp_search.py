import torch
from torch.utils.data import DataLoader
import random
import json
from util import (RiskDataset, device, get_data, 
                 best_params_name, model_name, get_criterion, get_optimizer, get_model)




def random_search(X_train, y_train, X_val, y_val, n_trials=10):
    # Define hyperparameter search space
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'hidden_size': [32, 64, 128, 256],
        'batch_size': [16, 32, 64, 128],
        'num_epochs': [30, 50, 100]
    }
    
    best_val_loss = float('inf')
    best_params = None
    best_model = None
    results = []
    
    for trial in range(n_trials):
        # Randomly sample parameters
        params = {
            'learning_rate': random.choice(param_grid['learning_rate']),
            'hidden_size': random.choice(param_grid['hidden_size']),
            'batch_size': random.choice(param_grid['batch_size']),
            'num_epochs': random.choice(param_grid['num_epochs'])
        }
        
        # Create datasets and dataloaders
        train_dataset = RiskDataset(X_train, y_train)
        val_dataset = RiskDataset(X_val, y_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Create model with current hidden size
        model = get_model(params['hidden_size'])
        criterion = get_criterion()
        optimizer = get_optimizer(model, params['learning_rate'])
        
        # Train model
        print(f"\nTrial {trial + 1}/{n_trials}")
        print(f"Training with parameters: {params}")
        model, train_losses, val_losses = train(model, train_loader, val_loader, 
                                              criterion, optimizer, params['num_epochs'])
        
        final_val_loss = val_losses[-1]
        results.append((params, final_val_loss))
        
        # Update best model if current one is better
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_params = params
            best_model = model
            
        print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Print results
    print("\nRandom Search Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_model, best_params




def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model = model.to(device)  # Ensure model is on correct device
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
    
    return model, train_losses, val_losses

X_train, y_train, X_val, y_val = get_data()
# Replace the existing model creation and training code with:
best_model, best_params = random_search(X_train, y_train, X_val, y_val, n_trials=10)
# Save the best parameters to a file
with open(best_params_name, 'w') as f:
    json.dump(best_params, f, indent=4)
# Save the final model and parameters
torch.save(best_model.cpu().state_dict(), model_name)
best_model.to(device)  # Move back to GPU if needed


print(f"Best parameters saved as {best_params_name}")
print(f"Model saved as {model_name}")
