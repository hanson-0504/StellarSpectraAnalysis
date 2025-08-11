# torch_nn_random_search.py
import math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
random.seed(42)

# --- 1) Model builder ---

class MLP(nn.Module):
    def __init__(self, in_dim, units1=256, units2=256, dropout1=0.1, dropout2=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, units1), nn.ReLU(), nn.Dropout(dropout1),
            nn.Linear(units1, units2), nn.ReLU(), nn.Dropout(dropout2),
            nn.Linear(units2, 1)  # regression head
        )

    def forward(self, x):
        return self.net(x)

def build_nn_model(hp: dict, input_dim: int):
    """Mirror your Keras build using an hp dict."""
    return MLP(
        in_dim=input_dim,
        units1=hp["units1"],
        units2=hp["units2"],
        dropout1=hp["dropout1"],
        dropout2=hp["dropout2"],
    )

# --- 2) Utilities ---

def rmse_from_mse(mse: float) -> float:
    return float(math.sqrt(mse))

def make_loaders(X, y, batch_size=512, val_split=0.2):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    n = len(X)
    n_val = int(val_split * n)
    idx = torch.randperm(n)
    tr, va = idx[:-n_val], idx[-n_val:]

    dl_tr = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(TensorDataset(X[va], y[va]), batch_size=batch_size, shuffle=False)
    return dl_tr, dl_va

# --- 3) One training run (one set of hparams) ---

def train_once(X, y, input_dim, hp, *, max_epochs=50, patience=5, batch_size=512, device=None, verbose=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_nn_model(hp, input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    # ReduceLROnPlateau analogue to your Keras callback
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    dl_tr, dl_va = make_loaders(X, y, batch_size=batch_size, val_split=0.2)
    mse = nn.MSELoss()

    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += mse(pred, yb).item()
                n += 1
        val_loss /= max(n, 1)
        scheduler.step(val_loss)

        if verbose:
            print(f"epoch {epoch+1:02d} val_mse={val_loss:.6f} val_rmse={math.sqrt(val_loss):.6f}")

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rmse_from_mse(best_val)

# --- 4) Random search (stand-in for KerasTuner.RandomSearch) ---

def random_search_space():
    return {
        "units1": random.choice([64, 128, 192, 256, 320, 384, 448, 512]),
        "units2": random.choice([64, 128, 192, 256, 320, 384, 448, 512]),
        "dropout1": random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "dropout2": random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "lr":      10 ** random.uniform(-4, -2),  # log-uniform 1e-4..1e-2
    }

def train_neural_network(X, y, param_name, *, max_trials=15, seed=42, verbose=False):
    """Mirror your Keras function: returns (best_model, best_hps, val_rmse)."""
    rng = random.Random(seed)
    input_dim = X.shape[1]
    best = {"val_rmse": float("inf"), "model": None, "hps": None}

    for t in range(1, max_trials + 1):
        # sample hparams
        hp = random_search_space()
        # train once
        model, val_rmse = train_once(X, y, input_dim, hp, max_epochs=50, patience=5, batch_size=512, verbose=verbose)
        if verbose:
            print(f"[trial {t:02d}] rmse={val_rmse:.6f} hps={hp}")

        if val_rmse < best["val_rmse"]:
            best.update(val_rmse=val_rmse, model=model, hps=hp)

    return best["model"], best["hps"], best["val_rmse"]

# --- 5) Predict ---

def predict_with_nn(model, X, batch_size=4096, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(X[i:i+batch_size].to(device)).cpu().numpy())
    return np.vstack(out).squeeze(-1)