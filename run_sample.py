import torch
from models.CardioMamba import ECGMamba2Classifier
# Create dummy input with shape [16, 300, 12]
dummy_input = torch.randn(16, 300, 12)  # batch_size=16, sequence_length=300, feature_dim=12

model = ECGMamba2Classifier(num_classes=2)


with torch.no_grad():
    output = model(dummy_input)

print(output.shape)  # Check output shape
