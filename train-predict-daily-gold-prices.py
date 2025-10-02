import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("Starting the gold price prediction script...")

# Load data
print("Loading data...")
df = pd.read_csv('gold_prices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
print(f"Data loaded. Shape: {df.shape}")

# Split data into training and future periods
print("Splitting data into training and future periods...")
train_data = df[df['Date'] <= '2020-12-31']
future_data = df[df['Date'] > '2020-12-31']
print(f"Training data shape: {train_data.shape}")
print(f"Future data shape: {future_data.shape}")

# Prepare data
print("Preparing and scaling data...")
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data['Price'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

seq_length = 60  # Use 60 days of historical data to predict the next day
X_train, y_train = create_sequences(train_scaled, seq_length)
print(f"Sequences created. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

# Initialize the model
print("Initializing the LSTM model...")
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

# Model initialization
model = ImprovedLSTMModel(input_size=1, hidden_size=100, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
print("Starting training...")
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

print("Training completed.")

# Save the model
print("Saving the trained model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'loss': avg_loss,
}, 'gold_price_lstm_model.pth')
print("Model saved successfully.")

# Load the model (in practice, you might do this in a separate script)
print("Loading the saved model...")
model = ImprovedLSTMModel(input_size=1, hidden_size=100, num_layers=2, output_size=1)
checkpoint = torch.load('gold_price_lstm_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully. Ready for prediction.")

# Prepare future data for prediction
print("Preparing future data for T+1 prediction...")
future_scaled = scaler.transform(future_data['Price'].values.reshape(-1, 1))

# Predict future prices (T+1 prediction)
print("Performing T+1 predictions...")
predictions = []
with torch.no_grad():
    for i in range(len(future_scaled) - seq_length):
        input_sequence = torch.FloatTensor(future_scaled[i:i+seq_length]).unsqueeze(0)
        next_pred = model(input_sequence)
        predictions.append(next_pred.item())

# Inverse transform predictions
predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Evaluate predictions
print("Evaluating T+1 predictions...")
actual_future = future_data['Price'].values[seq_length:]
mae = np.mean(np.abs(actual_future - predictions_inv.flatten()))
mape = np.mean(np.abs((actual_future - predictions_inv.flatten()) / actual_future)) * 100
print(f"MAE for T+1 predictions: {mae:.2f}")
print(f"MAPE for T+1 predictions: {mape:.2f}%")

# Visualize results
print("Generating visualization...")
plt.figure(figsize=(15, 6))
plt.plot(future_data['Date'][seq_length:], actual_future, label='Actual')
plt.plot(future_data['Date'][seq_length:], predictions_inv, label='Predicted (T+1)')
plt.title('Daily Gold Price T+1 Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

print("Script execution completed.")