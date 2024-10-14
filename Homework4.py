import pandas as pd
import torch
if torch.cuda.is_available():
  device = torch.device("cuda")
  print("cuda available");
else:
  device = torch.device("cpu")
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


df = pd.read_csv("mlp_regression_data.csv") 
df.head()



import matplotlib.pyplot as plt



X = df['x'].values.reshape(-1, 1)  
y = df['y'].values


print("X shape:", X.shape)
print("y shape:", y.shape)

print("Training size:", X.shape)


import numpy as np

#print("Training labels:", np.bincount(y_train))
#print("Validation labels:", np.bincount(y_val))
#print("Test labels:", np.bincount(y_test))

print("Training labels:", np.bincount(y.astype(int)))


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
                
            # 1st hidden layer
            torch.nn.Linear(num_features, 500),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(500, 300),
            torch.nn.ReLU(),
            # 3rd hidden layer
            torch.nn.Linear(300, 100),
            torch.nn.ReLU(),
            
            # output layer
            torch.nn.Linear(100,1),
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):

        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]        
        return x, y

    def __len__(self):
        return self.labels.shape[0]

train_ds = MyDataset(X, y)


train_loader = DataLoader(
    dataset=train_ds,
    batch_size=200,
    shuffle=True,
)


def compute_accuracy(model, dataloader):

    model.eval()
    mse_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for features, labels in dataloader:
            predictions = model(features)
            labels =labels.view(-1,1)
            mse_loss += F.mse_loss(predictions, labels, reduction='sum').item()
            total_examples += len(labels)
    return mse_loss / total_examples



torch.manual_seed(123)
model = MLP(num_features=1, num_classes=1)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # Stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 500

training_losses = []


for epoch in range(num_epochs):
    
    model = model.train()
    epoch_loss = 0 
    for features, labels in train_loader:

        logits = model(features)
        labels =labels.view(-1,1)
        loss = F.mse_loss(logits, labels) # Loss function
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    training_losses.append(avg_epoch_loss)
    train_acc = compute_accuracy(model, train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train ACC: {train_acc:.4f}")


from matplotlib.colors import ListedColormap
import numpy as np

def plot_fitted_line(X, y, model):
    plt.scatter(X, y, color='blue', label='Data points')
    X_tensor= torch.arange(X.min(), X.max(), 0.01).view(-1, 1)
    X_tensor = X_tensor.view(-1,1)
    #X_tensor = torch.tensor(np.linspace(X.min(), X.max(), 100).reshape(-1, 1), dtype=torch.float32)
    y_pred = model(X_tensor).detach().numpy()
    plt.plot(X_tensor.numpy(), y_pred, color='red', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_loss(epochCount,losses):
    epochs = range(1, epochCount + 1)
    plt.plot(epochs, losses, label='Training Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.show()

plot_fitted_line(X, y, model)

plot_loss(num_epochs,training_losses)
