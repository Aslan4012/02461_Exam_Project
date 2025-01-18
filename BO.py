device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective_function(params):
    learning_rate = params[0][0]
    batch_size = int(params[0][1])
    dropout_rate = params[0][2]


    print(f"Testing hyperparameters - Learning rate: {learning_rate}, Batch size: {batch_size}, Dropout rate: {dropout_rate}")

    model = CNN()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Use a subset of the training data (1%)
    train_subset_size = int(len(train_dataset) * 0.01)
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # Use a subset of the validation data (1%)
    val_subset_size = int(len(val_dataset) * 0.01)
    val_subset, _ = torch.utils.data.random_split(val_dataset, [val_subset_size, len(val_dataset) - val_subset_size])
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience = 3
    no_improve_epochs = 0

    for epoch in range(5):  
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.float32)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Validation Loss: {best_val_loss:.4f}")
    return best_val_loss

# Defining the bounds for the hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)}
]

#Bayesian Optimization object
optimizer = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=bounds)

# Initialize lists to track best parameters and validation loss across all iterations
best_parameters_all = []
best_validation_loss_all = []

#optimization process
max_iter = 10  #max iterations for optimization
optimizer.run_optimization(max_iter=max_iter)  


best_parameters = optimizer.X[np.argmin(optimizer.Y)]  # Best parameters
best_validation_loss = np.min(optimizer.Y)  # Best validationloss

print("\nOverall Best Hyperparameters after all iterations:")
print(f"Best hyperparameters: {best_parameters}")
print(f"Best validation loss: {best_validation_loss}")
