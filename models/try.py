import sys
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from models.blip import blip_feature_extractor  
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "checkpoints\model_base_14M.pth"

model = blip_feature_extractor(pretrained=model_path, image_size=224, vit="base")
model.eval()

coco = load_dataset("HuggingFaceM4/COCO", split="val", trust_remote_code=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_intrinsic_dimensionality(embeddings, explained_variance_threshold=0.95):
    
    if len(embeddings.shape) == 3:
        embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])
    else:
        embeddings_flat = embeddings
    
    embeddings_np = embeddings_flat.detach().cpu().numpy()
    pca = PCA().fit(embeddings_np)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    intrinsic_dim = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
    
    return intrinsic_dim

def analyze_clustering(embeddings, n_clusters=10):
    """
    Analyze clustering structure of embeddings
    """
    # Convert to numpy for sklearn
    embeddings_np = embeddings.cpu().numpy()
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    
    # Compute silhouette score
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(embeddings_np, cluster_labels)
    
    # Compute cluster sizes and densities
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))
    
    # Compute average distance to cluster center
    distances = []
    for i in range(n_clusters):
        cluster_points = embeddings_np[cluster_labels == i]
        if len(cluster_points) > 0:
            center = kmeans.cluster_centers_[i]
            dist = np.mean(cdist(cluster_points, [center]))
            distances.append(dist)
    
    return {
        'silhouette_score': silhouette_avg,
        'cluster_sizes': cluster_sizes,
        'avg_cluster_distance': np.mean(distances)
    }

def analyze_nearest_neighbors(embeddings, k=5):
    """
    Analyze nearest neighbor structure of embeddings
    """
    # Convert to numpy for sklearn
    embeddings_np = embeddings.cpu().numpy()
    
    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings_np)
    
    # Get distances and indices of nearest neighbors
    distances, indices = nbrs.kneighbors(embeddings_np)
    
    # Compute average distance to kth nearest neighbor
    avg_kth_distance = np.mean(distances[:, -1])
    
    # Compute local density (inverse of average distance to k nearest neighbors)
    local_density = 1 / np.mean(distances[:, 1:], axis=1)
    
    return {
        'avg_kth_distance': avg_kth_distance,
        'local_density_mean': np.mean(local_density),
        'local_density_std': np.std(local_density)
    }


num_samples = 100
all_text_embeddings = []
all_image_embeddings = []
all_q_embeddings = []
all_k_embeddings = []
all_v_embeddings = []

with torch.no_grad():
    for i, sample in tqdm(enumerate(coco), total=num_samples):
        
        if i >= num_samples:
            break
        
        # Process the image
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        images = transform(image).unsqueeze(0).to(device)
        
        # Get caption - COCO has multiple captions per image, we'll just use the first one
        caption = sample['captions'][0] if isinstance(sample['captions'], list) else sample['captions']
        
        multimodal_output = model(images, caption, mode='multimodal')
        multimodal_embeds, text_encoder_last, image_encoder_last, q, k, v = multimodal_output
        
        all_text_embeddings.append(text_encoder_last.cpu().clone())
        all_image_embeddings.append(image_encoder_last.cpu().clone())
        all_q_embeddings.append(q[0].cpu().clone())
        all_k_embeddings.append(k[0].cpu().clone())
        all_v_embeddings.append(v[0].cpu().clone())
    
# Stack all embeddings
stacked_text_embeddings = torch.cat(all_text_embeddings, dim=0)
stacked_image_embeddings = torch.cat(all_image_embeddings, dim=0)
stacked_q_embeddings = torch.cat(all_q_embeddings, dim=0)
stacked_k_embeddings = torch.cat(all_k_embeddings, dim=0)
stacked_v_embeddings = torch.cat(all_v_embeddings, dim=0)

print("text_shape", text_encoder_last.shape)
print("image_shape", image_encoder_last.shape)
print("q_shape", q[0].shape)
print("k_shape", k[0].shape)
print("v_shape", v[0].shape)

print("--------------------------------")
print("\nCalculating intrinsic dimensionality on the full dataset...")
   
text_intrinsic_dim = calculate_intrinsic_dimensionality(stacked_text_embeddings)
image_intrinsic_dim = calculate_intrinsic_dimensionality(stacked_image_embeddings)
q_intrinsic_dim = calculate_intrinsic_dimensionality(stacked_q_embeddings)
k_intrinsic_dim = calculate_intrinsic_dimensionality(stacked_k_embeddings)
v_intrinsic_dim = calculate_intrinsic_dimensionality(stacked_v_embeddings)

print("text_intrinsic_dim: ", text_intrinsic_dim)
print("image_intrinsic_dim: ", image_intrinsic_dim)
print("q_intrinsic_dim: ", q_intrinsic_dim)
print("k_intrinsic_dim: ", k_intrinsic_dim)
print("v_intrinsic_dim: ", v_intrinsic_dim)

# Optional - run clustering and nearest neighbor analysis
print("\n--- Running additional analyses ---")
print("Clustering analysis...")
text_clustering = analyze_clustering(stacked_text_embeddings)
image_clustering = analyze_clustering(stacked_image_embeddings)
print(f"Text silhouette score: {text_clustering['silhouette_score']:.4f}")
print(f"Image silhouette score: {image_clustering['silhouette_score']:.4f}")

print("\nNearest neighbor analysis...")
text_nn = analyze_nearest_neighbors(stacked_text_embeddings)
image_nn = analyze_nearest_neighbors(stacked_image_embeddings)
print(f"Text avg 5-NN distance: {text_nn['avg_kth_distance']:.4f}")
print(f"Image avg 5-NN distance: {image_nn['avg_kth_distance']:.4f}")
