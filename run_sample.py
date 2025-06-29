import torch
from models.CardioMamba import ECGMambaClassifier
# Create dummy input with shape [16, 300, 12]
dummy_input = torch.randn(16, 300, 12)  # batch_size=16, sequence_length=300, feature_dim=12

model = ECGMambaClassifier()


with torch.no_grad():
    output = model(dummy_input)

print(output.shape)  # Check output shape
