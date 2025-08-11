from types import SimpleNamespace
import train_rf_model as rf
import train_nn_model_v1 as nn

def train_rf(**kw):
    if hasattr(rf, 'run'):  return rf.run(**kw)
    return rf.train_and_save_models

def train_nn(**kw):
    if hasattr(nn, 'run'):  return nn.run(**kw)
    raise RuntimeError("No NN entrypoint found")
    
def train(strategy="auto", large_threshold=5000, **kw):
    # Heuristic (reuse your choose_model or a light version here)
    from agents.tools import choose_model, _dataset_profile, _has_gpu
    if strategy not in {'rf', 'nn'}:
        n, _ = _dataset_profile()
        strategy = "nn" if (n >= large_threshold and _has_gpu()) else "rf"
        return train_rf(**kw) if strategy=="rf" else train_nn(**kw)
