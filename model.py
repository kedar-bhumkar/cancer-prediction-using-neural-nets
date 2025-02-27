import torch.nn as nn


# ---------------------
# MODEL DEFINITION
# ---------------------

class RiskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RiskModel, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
        # Note: We won't explicitly apply SoftMax here because
        #       nn.CrossEntropyLoss does it internally.

    def forward(self, x):
        # Forward pass
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # The final output are raw logits (unnormalized scores).
        return out
