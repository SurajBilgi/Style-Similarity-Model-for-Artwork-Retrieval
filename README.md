# Style Similarity Model for Artwork Retrieval

This project implements a style similarity search system for artworks using CLIP (Contrastive Language-Image Pre-Training) from Hugging Face. The system can identify visually similar artworks and provides multiple interfaces for interaction.

## Features

- **CLIP-based Embeddings**: Uses OpenAI's CLIP model via Hugging Face for high-quality image embeddings
- **FAISS Vector Database**: Efficient similarity search using Facebook's FAISS library
- **Multiple Interfaces**: 
  - Jupyter Notebook for experimentation
  - Python script for batch processing
  - Streamlit web app for interactive use
- **Top-K Similar Images**: Returns the most similar artworks with confidence scores
- **Visual Results**: Display query and results with similarity scores

## Project Structure

```
Style-Similarity-Model-for-Artwork-Retrieval/
├── clip_artwork_similarity.ipynb    # Jupyter notebook
├── clip_artwork_similarity.py       # Python script
├── streamlit_app.py                 # Streamlit web application
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── data/                           # Place your artwork images here
├── uploaded_images/                # Streamlit uploaded images (auto-created)
├── image_database.pkl              # FAISS metadata (auto-created)
└── image_index.faiss              # FAISS index (auto-created)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Style-Similarity-Model-for-Artwork-Retrieval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Jupyter Notebook (`clip_artwork_similarity.ipynb`)

Best for experimentation and learning:

1. Place your artwork images in the `data/` directory
2. Open the notebook in Jupyter Lab/Notebook
3. Run cells sequentially
4. Modify the query image path in the last cell to test similarity search

### 2. Python Script (`clip_artwork_similarity.py`)

Best for batch processing:

1. Place your artwork images in the `data/` directory
2. Run the script:
```bash
python clip_artwork_similarity.py
```
3. Uncomment and modify the example usage section to test with your images

### 3. Streamlit Web App (`streamlit_app.py`)

Best for interactive use and demos:

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your browser to the provided URL (usually `http://localhost:8501`)

3. Use the web interface:
   - **Build Database Tab**: Upload multiple images to build your artwork database
   - **Search Tab**: Upload a query image to find similar artworks

## How It Works

### Architecture

1. **Image Preprocessing**: Images are loaded and converted to RGB format
2. **Feature Extraction**: CLIP model generates 512-dimensional embeddings
3. **Normalization**: Embeddings are L2-normalized for cosine similarity
4. **Indexing**: FAISS creates an efficient search index
5. **Similarity Search**: Cosine similarity finds the most similar images
6. **Results**: Top-K results returned with confidence scores

### CLIP Model

- **Model**: `openai/clip-vit-base-patch16`
- **Architecture**: Vision Transformer (ViT) base model
- **Embedding Size**: 512 dimensions
- **Similarity Metric**: Cosine similarity (via inner product on normalized vectors)

### FAISS Integration

- **Index Type**: `IndexFlatIP` (Inner Product for cosine similarity)
- **Search**: Exact nearest neighbor search
- **Storage**: Persistent storage with pickle for metadata

## Example Usage

### Notebook/Script
```python
# Generate embeddings for all images
embeddings = generate_embeddings(image_paths, processor, model, device)

# Find similar images
results = find_similar('data/query.jpg', top_k=5)

# Display results
show_results('data/query.jpg', results)
```

### Streamlit App
1. Upload artwork images in the "Build Database" tab
2. Switch to "Search Similar Images" tab
3. Upload a query image
4. Click "Search Similar Images"
5. View top 5 results with confidence scores

## Dependencies

- `torch`: PyTorch for deep learning
- `transformers`: Hugging Face transformers for CLIP
- `Pillow`: Image processing
- `numpy`: Numerical operations
- `faiss-cpu`: Vector similarity search
- `matplotlib`: Visualization (notebook/script)
- `tqdm`: Progress bars
- `streamlit`: Web application framework

## Performance Notes

- **GPU Support**: Automatically uses CUDA if available
- **Model Caching**: Streamlit caches the CLIP model for faster subsequent loads
- **Memory Usage**: Scales with number of images in database
- **Search Speed**: Sub-second similarity search for thousands of images

## Scaling Considerations

For production deployment, consider:

- **Distributed Storage**: Use cloud storage for images and embeddings
- **Database Scaling**: Consider Milvus or Pinecone for larger datasets
- **Batch Processing**: Process images in batches for memory efficiency
- **API Integration**: Convert to REST API for service integration
- **Monitoring**: Add logging and performance monitoring

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Missing Images**: Ensure images are in correct directory
3. **Model Download**: First run requires internet for model download
4. **File Permissions**: Ensure write permissions for database files

**Supported Image Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.