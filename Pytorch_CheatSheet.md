# PyTorch Cheat Sheet (LearnPyTorch.io Style)

```python
# 1. Imports & Setup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from pathlib import Path

# 2. Tensor Creation
torch.tensor(data)                  # Create tensor from Python data
torch.zeros(size=(d1, d2))          # Tensor filled with zeros
torch.ones(size=(d1, d2))           # Tensor filled with ones
torch.rand(size=(d1, d2))           # Random tensor (uniform distribution 0–1)
torch.randn(size=(d1, d2))          # Random tensor (normal distribution)

# 3. Tensor Operations
tensor1 + tensor2                   # Element-wise addition
tensor1 - tensor2                   # Element-wise subtraction
tensor1 * tensor2                   # Element-wise multiplication
tensor1 / tensor2                   # Element-wise division
torch.matmul(tensor1, tensor2)      # Matrix multiplication; torch.mm; called as dot product
                                    # and '@' can also be used for ex: tensor1 @ tensor2
tensor.reshape(new_shape)           # Reshape tensor
tensor.view(new_shape)              # Another way to reshape tensor
tensor.T                            # Transpose tensor
tensor.squeeze()                    # Remove dimensions of size 1
tensor.unsqueeze(dim)               # Add dimension at index

# 4. Tensor Info
tensor.shape                        # Returns tensor shape
tensor.ndim                         # Number of dimensions
tensor.dtype                        # Data type of tensor
tensor.device                       # Shows which device tensor is on

# 5. GPU / Device
torch.cuda.is_available()           # Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor.to(device)                   # Move tensor to GPU or CPU
model.to(device)                    # Move model to GPU or CPU

# 6. Neural Network Modules
nn.Linear(in_features, out_features)    # Fully connected layer
nn.Conv2d(in_channels, out_channels, kernel_size)  # 2D convolution layer
nn.ReLU()                               # ReLU activation function
nn.Sigmoid()                            # Sigmoid activation
nn.Flatten()                            # Flatten tensor
nn.Sequential(layer1, layer2, ...)      # Stack layers in order

# 7. Loss Functions
nn.MSELoss()                            # Mean squared error loss
nn.CrossEntropyLoss()                   # Classification loss

# 8. Optimizers
torch.optim.SGD(params, lr=0.01)        # Stochastic Gradient Descent
torch.optim.Adam(params, lr=0.001)      # Adam optimizer

# 9. Training Loop
model.train()                           # Set training mode
optimizer.zero_grad()                   # Clear gradients
outputs = model(X)                      # Forward pass
loss = loss_fn(outputs, y)              # Compute loss
loss.backward()                         # Backpropagation
optimizer.step()                        # Update model weights

# 10. Evaluation
model.eval()                            # Set evaluation mode
with torch.no_grad():                   # Disable gradients
    preds = model(X_test)               # Make predictions

# 11. Data Loading
DataLoader(dataset, batch_size=B, shuffle=True)  # Loads dataset in batches

# 12. Saving and Loading Models
torch.save(model.state_dict(), "model.pth")      # Save model weights
model.load_state_dict(torch.load("model.pth"))   # Load model weights

# 13. Transfer Learning
weights = torchvision.models.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights)
for param in model.parameters():
    param.requires_grad = False                # Freeze base model
model.fc = nn.Linear(in_features=2048, out_features=num_classes)  # Replace final layer

# 14. PyTorch 2.0
torch.compile(model)                          # Compile model for speed
torch.set_default_device("cuda")              # Set GPU as default device