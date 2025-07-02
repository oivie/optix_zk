# Simple MLP for predicting light scatter result
import torch
import torch.nn as nn
import numpy as np

class LightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

def train():
    X = np.random.rand(100, 3).astype(np.float32)
    y = np.sum(X, axis=1, keepdims=True).astype(np.float32)  # dummy target

    model = LightNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    for epoch in range(300):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "../models/light_model.pth")
    torch.onnx.export(model, X_t[:1], "../models/light_model.onnx", input_names=["input"], output_names=["output"])
    print("Model trained and exported to ONNX.")

if __name__ == "__main__":
    train()
