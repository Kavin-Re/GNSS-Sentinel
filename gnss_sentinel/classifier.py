"""Small softmax classifier over window features (style follows FHSS-Link/sim/classify.py). Trained on synthetic data
only; its output is a *hint* for the operator, not evidence."""
import json
import numpy as np


def train(X, y, k, iters=3000, lr=0.5, l2=1e-3):
    X = np.asarray(X, float); mu, sd = X.mean(0), X.std(0) + 1e-9
    Xb = np.c_[(X - mu) / sd, np.ones(len(X))]; W = np.zeros((Xb.shape[1], k)); Y = np.eye(k)[y]
    for _ in range(iters):
        Z = Xb @ W; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (Xb.T @ (P - Y) / len(Xb) + l2 * W)
    return dict(mu=mu.tolist(), sd=sd.tolist(), W=W.tolist())


def proba(model, X):
    X = np.atleast_2d(X)
    Z = np.c_[(X - np.array(model["mu"])) / np.array(model["sd"]), np.ones(len(X))] @ np.array(model["W"])
    Z -= Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)


def predict(model, X):
    return proba(model, X).argmax(1)


def save(model, classes, path):
    with open(path, "w") as f: json.dump(dict(classes=classes, **model), f)


def load(path):
    with open(path) as f: d = json.load(f)
    return {k: d[k] for k in ("mu", "sd", "W")}, d["classes"]
