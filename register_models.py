#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# register_models_simple.py
from pathlib import Path
import json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
import mlflow, mlflow.pyfunc

MLFLOW_URI = "http://127.0.0.1:5000"
MODELS_DIR = Path("Models")
ART_DIR    = Path("artifacts_tf")
SEQ_LEN = 30

# LSTM hyperparams used during training (match your grid)
LSTM_CFG = {
    "artifact_1": dict(hidden=64,  latent=16, layers=1),
    "artifact_2": dict(hidden=96,  latent=24, layers=1),
    "artifact_3": dict(hidden=80,  latent=20, layers=1),
    "artifact_4": dict(hidden=64,  latent=16, layers=2),
    "artifact_5": dict(hidden=72,  latent=18, layers=1),
}

class DenseAE(nn.Module):
    def __init__(self, in_dim=60):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,40),nn.ReLU(),nn.Linear(40,20),nn.ReLU(),nn.Linear(20,10),nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(10,20),nn.ReLU(),nn.Linear(20,40),nn.ReLU(),nn.Linear(40,in_dim))
    def forward(self,x): return self.dec(self.enc(x))

class LSTMAE(nn.Module):
    def __init__(self,input_dim=60,hidden_dim=64,latent_dim=16,num_layers=1):
        super().__init__()
        self.encoder=nn.LSTM(input_dim,hidden_dim,num_layers=num_layers,batch_first=True)
        self.to_latent=nn.Linear(hidden_dim,latent_dim); self.from_latent=nn.Linear(latent_dim,hidden_dim)
        self.decoder=nn.LSTM(input_dim,hidden_dim,num_layers=num_layers,batch_first=True); self.out=nn.Linear(hidden_dim,input_dim)
        self.num_layers=num_layers; self.hidden_dim=hidden_dim
    def forward(self,x):
        enc,_=self.encoder(x); h=enc[:,-1,:]; z=self.to_latent(h); base=self.from_latent(z)
        h0=base.unsqueeze(0).repeat(self.num_layers,1,1)
        c0=torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        dec,_=self.decoder(x,(h0,c0)); return self.out(dec)

class DenseWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, ctx):
        self.features=json.loads(Path(ctx.artifacts["features"]).read_text())["features"]
        self.scaler=joblib.load(ctx.artifacts["scaler"])
        self.model=torch.jit.load(ctx.artifacts["dense_ts"]); self.model.eval()
    def predict(self, ctx, df: pd.DataFrame):
        X=df.reindex(columns=self.features, fill_value=0.0).astype(np.float32).values
        Xs=self.scaler.transform(X).astype(np.float32)
        with torch.no_grad(): x_hat=self.model(torch.from_numpy(Xs)).numpy()
        return pd.DataFrame({"dense_healthscore": ((Xs-x_hat)**2).mean(axis=1)})

class LSTMWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, ctx):
        self.features=json.loads(Path(ctx.artifacts["features"]).read_text())["features"]
        self.scaler=joblib.load(ctx.artifacts["scaler"])
        self.seq_len=int(Path(ctx.artifacts["seq_len_txt"]).read_text())
        self.model=torch.jit.load(ctx.artifacts["lstm_ts"]); self.model.eval()
    def predict(self, ctx, df: pd.DataFrame):
        X=df.reindex(columns=self.features, fill_value=0.0).astype(np.float32).values
        Xs=self.scaler.transform(X).astype(np.float32); n,d=Xs.shape; L=self.seq_len
        out=np.full((n,), np.nan, np.float32)
        if n>=L:
            seq=np.stack([Xs[i:i+L] for i in range(n-L+1)], axis=0)
            with torch.no_grad(): xh=self.model(torch.from_numpy(seq)).numpy()
            err=((seq-xh)**2).mean(axis=2)[:, -1]; out[L-1:]=err
        return pd.DataFrame({"lstm_healthscore": out})

def export_torchscript_dense(in_dim, weights, out_dir):
    m=DenseAE(in_dim); m.load_state_dict(torch.load(weights, map_location="cpu")); m.eval()
    ex=torch.randn(1,in_dim,dtype=torch.float32); ts=out_dir/"dense_ae.torchscript.pt"
    out_dir.mkdir(parents=True, exist_ok=True); torch.jit.trace(m,ex).save(str(ts)); return ts

def export_torchscript_lstm(in_dim, cfg, weights, out_dir, L):
    m=LSTMAE(input_dim=in_dim, **cfg); m.load_state_dict(torch.load(weights, map_location="cpu")); m.eval()
    ex=torch.randn(1,L,in_dim,dtype=torch.float32); ts=out_dir/"lstm_ae.torchscript.pt"
    out_dir.mkdir(parents=True, exist_ok=True); torch.jit.trace(m,ex).save(str(ts)); return ts

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    features = json.loads((ART_DIR/"features.json").read_text())["features"]; in_dim=len(features)
    scaler = ART_DIR/"scaler_prod.pkl"
    for sub in sorted(MODELS_DIR.glob("artifact_*")):
        aid=sub.name
        dense_pt=sub/"dense_ae.pt"; lstm_pt=sub/"lstm_ae.pt"
        if not dense_pt.exists() or not lstm_pt.exists(): continue
        exp=sub/"export"; dense_ts=export_torchscript_dense(in_dim, dense_pt, exp)
        lstm_ts=export_torchscript_lstm(in_dim, LSTM_CFG[aid], lstm_pt, exp, SEQ_LEN)

        # DENSE
        with mlflow.start_run(run_name=f"{aid}-dense"):
            mlflow.set_tags({"artifact_id":aid,"model_type":"dense"})
            logged=mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=DenseWrapper(),
                artifacts={"features": str(ART_DIR/"features.json"),
                           "scaler": str(scaler),
                           "dense_ts": str(dense_ts)},
                pip_requirements=["mlflow","pandas","numpy","torch","scikit-learn","joblib"],
                registered_model_name="signals-anomaly-ae-dense"
            )
        # LSTM
        with mlflow.start_run(run_name=f"{aid}-lstm"):
            mlflow.set_tags({"artifact_id":aid,"model_type":"lstm"})
            seq_len_txt = mlflow.pyfunc.model._save_artifact("seq_len.txt", data=str(SEQ_LEN).encode())
            logged=mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=LSTMWrapper(),
                artifacts={"features": str(ART_DIR/"features.json"),
                           "scaler": str(scaler),
                           "seq_len_txt": seq_len_txt,
                           "lstm_ts": str(lstm_ts)},
                pip_requirements=["mlflow","pandas","numpy","torch","scikit-learn","joblib"],
                registered_model_name="signals-anomaly-ae-lstm"
            )
    print("Registered all artifacts.")

if __name__=="__main__":
    main()

