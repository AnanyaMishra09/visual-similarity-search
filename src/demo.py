"""Gradio demo: upload a query image, get the top-5 most similar CIFAR-100
test images back, with class labels and similarity scores.

Run with:
    python demo.py --run-dir runs/baseline
It loads runs/baseline/best.pt and runs/baseline/test_embeddings.npz (both
written by train.py), plus a small cache of gallery thumbnails for display.
"""
import argparse
import os
import sys

import numpy as np
import torch
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from data import build_transforms, build_splits, get_cifar100, concat_cifar_pool, IndexedCIFAR100
from model import build_model
from retrieval import BruteForceIndex
from utils import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/baseline")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def build_app(args):
    device = get_device(args.device)

    model = build_model(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run_dir, "best.pt"), map_location=device))
    model.eval()

    npz = np.load(os.path.join(args.run_dir, "test_embeddings.npz"))
    gallery_emb, gallery_labels = npz["gallery_emb"], npz["gallery_labels"]
    index = BruteForceIndex(gallery_emb, gallery_labels)

    # Rebuild gallery image lookup (same deterministic split as training)
    # so we can render actual thumbnails for the retrieved results.
    train_raw, test_raw = get_cifar100(args.data_root)
    pool = concat_cifar_pool(train_raw, test_raw)
    splits = build_splits(pool.targets, seed=42)
    gallery_indices = splits["gallery"]
    class_names = pool.classes

    eval_tf = build_transforms(train=False)

    def embed_pil(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((32, 32))
        tensor = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor)
        return emb.cpu().numpy()

    def search(query_img: Image.Image, top_k: int):
        if query_img is None:
            return []
        q_emb = embed_pil(query_img)
        top_idx, top_dist = index.query(q_emb, k=int(top_k))
        top_idx, top_dist = top_idx[0], top_dist[0]

        gallery = []
        for rank, (i, d) in enumerate(zip(top_idx, top_dist)):
            orig_idx = int(gallery_indices[i])
            arr = pool.data[orig_idx]
            label = int(gallery_labels[i])
            similarity = max(0.0, 1.0 - (d ** 2) / 4.0)  # cosine-equivalent for unit vectors
            caption = f"#{rank+1}: {class_names[label]} (dist={d:.3f}, sim={similarity:.3f})"
            gallery.append((Image.fromarray(arr), caption))
        return gallery

    with gr.Blocks(title="Visual Similarity Search (CIFAR-100)") as demo:
        gr.Markdown(
            "# Visual Similarity Search\n"
            "Upload any image (ideally a simple centered object/animal, CIFAR-100 style). "
            "The model embeds it with a from-scratch ResNet-18 trained with batch-hard triplet "
            "loss, and retrieves the nearest CIFAR-100 test images by L2 distance in embedding space."
        )
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Image(type="pil", label="Query image")
                top_k_slider = gr.Slider(1, 10, value=args.top_k, step=1, label="Top-K")
                search_btn = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                results = gr.Gallery(label="Top-K most similar images", columns=5, height=300)

        search_btn.click(fn=search, inputs=[query_input, top_k_slider], outputs=results)
        query_input.change(fn=search, inputs=[query_input, top_k_slider], outputs=results)

    return demo


if __name__ == "__main__":
    args = parse_args()
    app = build_app(args)
    app.launch(share=args.share)
