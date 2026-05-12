"""
diagnose_gradients.py — quick diagnostic
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = "data"
FAMILIES   = ["kinase", "oxidoreductase", "transcription_factor",
              "chaperone", "transporter"]
N_PER_FAM  = 12
N_DIMS     = 320
HIDDEN_DIM = 64

def load_layer(layer):
    in_act  = np.load(f"{DATA_DIR}/in_dist_layer{layer}.npy")
    ood_act = np.load(f"{DATA_DIR}/ood_layer{layer}.npy")
    return in_act, ood_act

def get_envs(in_act, ood_act):
    envs = []
    for i in range(len(FAMILIES)):
        s = i * N_PER_FAM; e = s + N_PER_FAM
        x = np.vstack([in_act[s:e], ood_act[s:e]])
        y = np.array([1.0]*N_PER_FAM + [0.0]*N_PER_FAM)
        x = StandardScaler().fit_transform(x)
        envs.append((
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        ))
    return envs

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_DIMS, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

in_act, ood_act = load_layer(2)
envs = get_envs(in_act, ood_act)
model = Net()
criterion = nn.BCELoss()

# Check gradients BEFORE any training
print("=== Gradient check BEFORE training ===")
env_grads = []
model.train()
for i, (x_e, y_e) in enumerate(envs):
    x_req = x_e.clone().requires_grad_(True)
    loss  = criterion(model(x_req), y_e)
    grad  = torch.autograd.grad(loss, x_req)[0]
    gabs  = grad.abs().mean(dim=0).detach().numpy()
    env_grads.append(gabs)
    print(f"  env {i}: grad mean={gabs.mean():.6f}  max={gabs.max():.6f}")

grads_matrix = np.stack(env_grads, axis=0)
variance     = grads_matrix.var(axis=0)
print(f"\nVariance across envs: min={variance.min():.8f}  "
      f"max={variance.max():.8f}  mean={variance.mean():.8f}")

# Train for just 50 epochs (ERM only)
print("\n=== Training 50 epochs ERM only ===")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for t in range(50):
    model.train()
    optimizer.zero_grad()
    loss = torch.stack([criterion(model(x_e), y_e)
                        for x_e, y_e in envs]).mean()
    loss.backward()
    optimizer.step()
print(f"  Final ERM loss: {loss.item():.4f}")

# Check gradients AFTER ERM training
print("\n=== Gradient check AFTER 50 epochs ERM ===")
env_grads2 = []
model.train()
for i, (x_e, y_e) in enumerate(envs):
    x_req = x_e.clone().requires_grad_(True)
    loss  = criterion(model(x_req), y_e)
    grad  = torch.autograd.grad(loss, x_req)[0]
    gabs  = grad.abs().mean(dim=0).detach().numpy()
    env_grads2.append(gabs)
    print(f"  env {i}: grad mean={gabs.mean():.6f}  max={gabs.max():.6f}")

grads_matrix2 = np.stack(env_grads2, axis=0)
variance2     = grads_matrix2.var(axis=0)
print(f"\nVariance across envs: min={variance2.min():.8f}  "
      f"max={variance2.max():.8f}  mean={variance2.mean():.8f}")
print(f"Non-zero variance dims: {(variance2 > 1e-10).sum()} / {N_DIMS}")