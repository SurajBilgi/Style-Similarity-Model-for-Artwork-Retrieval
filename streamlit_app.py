import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import faiss
import pickle
from transformers import CLIPProcessor, CLIPModel
import tempfile
import shutil

# Configure Streamlit page
st.set_page_config(
    page_title="Artwork Similarity Search", page_icon="🎨", layout="wide"
)


@st.cache_resource
def load_clip_model():
    """Load CLIP model and processor with caching"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


def get_image_embedding(image, processor, model, device):
    """Generate CLIP embedding for a single image"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(
            **{k: v.to(device) for k, v in inputs.items()}
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()


def save_database(embeddings, image_paths, metadata_path="image_database.pkl"):
    """Save FAISS index and image metadata"""
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype("float32"))

    # Save metadata
    metadata = {"image_paths": image_paths, "embeddings": embeddings}

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    faiss.write_index(index, "image_index.faiss")
    return index


def load_database(metadata_path="image_database.pkl"):
    """Load FAISS index and image metadata"""
    try:
        # Check if both files exist before trying to read
        if not os.path.exists("image_index.faiss") or not os.path.exists(metadata_path):
            return None, [], np.array([])

        index = faiss.read_index("image_index.faiss")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata["image_paths"], metadata["embeddings"]
    except (FileNotFoundError, Exception):
        return None, [], np.array([])


def search_similar_images(query_embedding, index, image_paths, top_k=5):
    """Search for similar images using FAISS"""
    if index is None or len(image_paths) == 0:
        return []

    scores, indices = index.search(query_embedding.astype("float32"), top_k)

    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(image_paths):
            results.append((image_paths[idx], float(score)))

    return results


def main():
    st.title("🎨 Artwork Similarity Search")
    st.markdown(
        "Upload artwork images to build a database and search for similar artworks!"
    )

    # Load CLIP model
    processor, model, device = load_clip_model()

    # Load existing database
    index, image_paths, embeddings = load_database()

    st.sidebar.header("Database Status")
    st.sidebar.write(f"Images in database: {len(image_paths)}")

    # Create tabs
    tab1, tab2 = st.tabs(["🏗️ Build Database", "🔍 Search Similar Images"])

    with tab1:
        st.header("Upload Images to Build Database")

        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Process and Add to Database"):
            # Create temporary directory for uploaded images
            temp_dir = tempfile.mkdtemp()
            new_image_paths = []
            new_embeddings = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                for i, uploaded_file in enumerate(uploaded_files):
                    # Save uploaded file temporarily
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Process image
                    image = Image.open(temp_path).convert("RGB")
                    embedding = get_image_embedding(image, processor, model, device)

                    # Save to permanent location
                    permanent_dir = "uploaded_images"
                    os.makedirs(permanent_dir, exist_ok=True)
                    permanent_path = os.path.join(permanent_dir, uploaded_file.name)
                    shutil.copy2(temp_path, permanent_path)

                    new_image_paths.append(permanent_path)
                    new_embeddings.append(embedding[0])

                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}...")

                # Combine with existing database
                all_image_paths = image_paths + new_image_paths
                if len(embeddings) > 0:
                    all_embeddings = np.vstack([embeddings, np.array(new_embeddings)])
                else:
                    all_embeddings = np.array(new_embeddings)

                # Save updated database
                index = save_database(all_embeddings, all_image_paths)

                st.success(f"✅ Added {len(uploaded_files)} images to database!")
                st.rerun()

            except Exception as e:
                st.error(f"Error processing images: {str(e)}")
            finally:
                # Cleanup temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                progress_bar.empty()
                status_text.empty()

    with tab2:
        st.header("Search for Similar Images")

        if len(image_paths) == 0:
            st.warning("⚠️ No images in database. Please upload images first!")
            return

        query_file = st.file_uploader(
            "Upload a query image", type=["png", "jpg", "jpeg"], key="query_image"
        )

        if query_file is not None:
            # Display query image
            query_image = Image.open(query_file).convert("RGB")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Query Image")
                st.image(query_image, caption="Query", use_column_width=True)

                # Number of results slider
                top_k = st.slider("Number of results", 1, 10, 5)

                search_button = st.button("🔍 Search Similar Images")

            with col2:
                if search_button:
                    with st.spinner("Searching for similar images..."):
                        # Get query embedding
                        query_embedding = get_image_embedding(
                            query_image, processor, model, device
                        )

                        # Search similar images
                        results = search_similar_images(
                            query_embedding, index, image_paths, top_k
                        )

                        if results:
                            st.subheader(f"Top {len(results)} Similar Images")

                            # Display results in a grid
                            cols = st.columns(min(3, len(results)))

                            for i, (image_path, score) in enumerate(results):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    if os.path.exists(image_path):
                                        result_image = Image.open(image_path)
                                        st.image(
                                            result_image,
                                            caption=f"Score: {score:.3f}",
                                            use_column_width=True,
                                        )
                                        st.write(f"📁 {os.path.basename(image_path)}")
                                    else:
                                        st.error(f"Image not found: {image_path}")
                        else:
                            st.warning("No similar images found.")

        # Display database images
        if st.checkbox("Show all images in database"):
            st.subheader("Database Images")
            if image_paths:
                cols = st.columns(5)
                for i, image_path in enumerate(image_paths):
                    col_idx = i % 5
                    with cols[col_idx]:
                        if os.path.exists(image_path):
                            img = Image.open(image_path)
                            st.image(
                                img,
                                caption=os.path.basename(image_path),
                                use_column_width=True,
                            )


if __name__ == "__main__":
    main()
