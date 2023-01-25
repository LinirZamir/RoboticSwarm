import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model = torch.nn.Linear(1, 1)

# Set the model to run on the specified device
model = model.to(device)

# Define the loss function and the optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Generate some synthetic data
X = torch.randn(100, 1).to(device)
y = 2 * X + 3 + torch.randn(100, 1).to(device)

# Run 1000 training iterations
for i in range(1000):
    # Forward pass: compute the predicted y values
    y_pred = model(X)
    
    # Compute the loss
    loss = loss_fn(y_pred, y)
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Backward pass: compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print the loss every 100 iterations
    if i % 100 == 0:
        print(f'Loss at iteration {i}: {loss.item():.4f}')

# Check the model's predictions
X_test = torch.randn(1, 1).to(device)
y_test = model