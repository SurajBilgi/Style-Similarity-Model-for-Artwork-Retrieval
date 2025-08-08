import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel


def load_model(device):
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def get_image_paths(image_dir):
    return [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith((".jpg", ".png"))
    ]


def generate_embeddings(image_paths, processor, model, device):
    embeddings = []
    for path in tqdm(image_paths, desc="Embedding images"):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(
                **{k: v.to(device) for k, v in inputs.items()}
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())
    return np.vstack(embeddings)


def find_similar(
    query_path, embeddings, image_paths, processor, model, device, top_k=5
):
    image = Image.open(query_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        query_features = model.get_image_features(
            **{k: v.to(device) for k, v in inputs.items()}
        )
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    sims = (embeddings @ query_features.cpu().numpy().T).squeeze()
    top_indices = np.argsort(-sims)[:top_k]
    return [(image_paths[i], float(sims[i])) for i in top_indices]


def show_results(query_path, results):
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 5))
    axes[0].imshow(Image.open(query_path))
    axes[0].set_title("Query")
    axes[0].axis("off")
    for i, (path, score) in enumerate(results):
        axes[i + 1].imshow(Image.open(path))
        axes[i + 1].set_title(f"Score: {score:.2f}")
        axes[i + 1].axis("off")
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir = "data"
    image_paths = get_image_paths(image_dir)
    print(f"Found {len(image_paths)} images.")
    processor, model = load_model(device)
    embeddings = generate_embeddings(image_paths, processor, model, device)
    # Example usage:
    # query_img = 'data/query.jpg'
    # results = find_similar(query_img, embeddings, image_paths, processor, model, device)
    # show_results(query_img, results)
