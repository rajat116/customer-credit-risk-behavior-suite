# src/train_autoencoder.py
from pathlib import Path
import json
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .config import DATA_PROCESSED, MODELS_DIR
from .utils import load_csv, load_scaler


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def main():
    df = load_csv(DATA_PROCESSED)
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # Select features
    X = df[feature_cols].copy()

    # FIX: impute missing values
    X = X.fillna(X.median())

    # Convert to numpy and scale
    X = X.values
    scaler = load_scaler("scaler_pd.pkl")
    X_s = scaler.transform(X)

    # Convert to tensor
    X_tensor = torch.tensor(X_s, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Model
    input_dim = X_s.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=8)

    # FIX: smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()

            recon, _ = model(batch)
            loss = criterion(recon, batch)

            loss.backward()

            # FIX: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        print(f"Epoch {epoch+1}, loss={total_loss / len(dataset):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    print("Saved autoencoder to models/autoencoder.pt")


if __name__ == "__main__":
    main()