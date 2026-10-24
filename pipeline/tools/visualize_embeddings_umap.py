#!/usr/bin/env python3
"""
visualize_embeddings_umap_all.py
--------------------------------
Loads all .npz embedding shards, merges them, and visualizes with UMAP + Plotly + Dash.
Clicking a point displays the corresponding album cover image.
"""

import os
import glob
import numpy as np
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output
import umap

from embed_batch import OUT_DIR, IMAGE_ROOT

# ---- CONFIG ----
RANDOM_SEED = 42
N_NEIGHBORS = 15
MIN_DIST = 0.1
MAX_POINTS = None  # or set to an int if you want a random subsample for speed

# ---- LOAD AND MERGE SHARDS ----
shards = sorted(glob.glob(os.path.join(OUT_DIR, "*.npz")))
if not shards:
    raise FileNotFoundError("No .npz files found in directory.")

all_embs = []
all_uuids = []

print(f"Found {len(shards)} shards. Loading and merging...")
for fpath in shards:
    data = np.load(fpath, allow_pickle=True)
    embs = data["embs"].astype(np.float32)
    uuids = data["uuids"]
    all_embs.append(embs)
    all_uuids.append(uuids)
    print(f"Loaded {fpath}: {len(embs)} embeddings")

embs = np.concatenate(all_embs, axis=0)
uuids = np.concatenate(all_uuids, axis=0)
print(f"Total merged: {len(embs):,} embeddings")

if MAX_POINTS and len(embs) > MAX_POINTS:
    np.random.seed(RANDOM_SEED)
    idx = np.random.choice(len(embs), MAX_POINTS, replace=False)
    embs = embs[idx]
    uuids = uuids[idx]
    print(f"Subsampled to {len(embs):,} points for performance.")

# ---- UMAP DIMENSIONALITY REDUCTION ----
print("Running UMAP (3D projection)...")
umap_model = umap.UMAP(
    n_components=3,
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    metric="cosine",
    random_state=RANDOM_SEED
)
coords = umap_model.fit_transform(embs)
print("UMAP complete.")

# ---- BUILD DASH APP ----
app = Dash(__name__)

scatter = go.Scatter3d(
    x=coords[:,0],
    y=coords[:,1],
    z=coords[:,2],
    mode="markers",
    marker=dict(size=3, color=coords[:,2], colorscale="Viridis", opacity=0.7),
    text=uuids,
    hovertemplate="UUID: %{text}<extra></extra>",
)

fig = go.Figure(data=[scatter])
fig.update_layout(
    title="UMAP Embedding Visualization (click a point to view image)",
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        zaxis_title="UMAP 3"
    )
)

app.layout = html.Div([
    html.Div([
        html.Img(id="album-img", style={"height": "256px", "border": "2px solid #444"}),
    ], style={"position": "absolute", "top": "20px", "left": "20px", "zIndex": "10"}),
    dcc.Graph(id="embedding-plot", figure=fig, style={"height": "95vh"})
])

@app.callback(
    Output("album-img", "src"),
    Input("embedding-plot", "clickData")
)
def display_image(clickData):
    if not clickData:
        return None
    uuid = clickData["points"][0]["text"]
    prefix = uuid[:2]
    img_path = os.path.join(IMAGE_ROOT, prefix, f"{uuid}.jpg")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return None
    import base64
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

if __name__ == "__main__":
    print("Launching interactive visualization at http://127.0.0.1:8050 ...")
    app.run(debug=True, use_reloader=False)
