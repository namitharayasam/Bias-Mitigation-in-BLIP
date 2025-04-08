import sys
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from models.blip import blip_feature_extractor  
from tqdm import tqdm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "checkpoints/model_base_14M.pth"

model = blip_feature_extractor(pretrained=model_path, image_size=224, vit="base").to(device)
model.eval()

fairface = load_dataset("HuggingFaceM4/FairFace", "0.25", split="train", trust_remote_code=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def rbf_kernel(X, Y, sigma=1.0):
    """Computes the RBF (Gaussian) kernel matrix."""
    XX = torch.cdist(X, X, p=2).pow(2)  # ||x_i - x_j||^2
    YY = torch.cdist(Y, Y, p=2).pow(2)  # ||y_i - y_j||^2
    XY = torch.cdist(X, Y, p=2).pow(2)  # ||x_i - y_j||^2

    return torch.exp(-XX / (2 * sigma ** 2)), torch.exp(-YY / (2 * sigma ** 2)), torch.exp(-XY / (2 * sigma ** 2))

def mmd(X, Y, sigma=1.0):
    """Computes the squared Maximum Mean Discrepancy (MMD) using the RBF kernel."""
    K_XX, K_YY, K_XY = rbf_kernel(X, Y, sigma)
    m, n = X.shape[0], Y.shape[0]

    mmd_squared = K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n)
    return mmd_squared.sqrt()  # Take sqrt to get actual MMD

def linear_CKA(X, Y):
    """
    Compute the linear CKA between X and Y.
    X, Y: tensors of shape (n_samples, n_features)
    Returns a scalar similarity value in [0,1] (1 means perfect alignment).
    """
    # Center X and Y
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)
    # Compute Gram (kernel) matrices
    K = torch.mm(X_centered, X_centered.t())
    L = torch.mm(Y_centered, Y_centered.t())
    # Compute HSIC (Frobenius inner product between centered Gram matrices)
    hsic = torch.norm(torch.mm(X_centered.t(), Y_centered), p='fro')**2
    # Alternatively, for linear kernel, one can compute:
    # hsic = (K * L).sum()
    # Normalize by the self-similarities:
    norm_x = torch.norm(K, p='fro')**2
    norm_y = torch.norm(L, p='fro')**2
    return hsic / (torch.sqrt(norm_x * norm_y) + 1e-10)

mmd_text_q_list = []
mmd_image_k_list = []
mmd_image_v_list = []

cka_text_q_list = []
cka_image_k_list = []
cka_image_v_list = []

for i, sample in enumerate(fairface):

    images = transform(sample['image']).unsqueeze(0).to(device)
    captions = f"A {sample['age']} year old {sample['gender']} with {sample['race']} ethnicity."

    with torch.no_grad():
        multimodal_features, text_encoder_last, image_encoder_last, q, k, v = model(images, captions, mode="multimodal")

    # print(len(q))
    # Compute MMD distances
    mmd_text_q_list.append(mmd(text_encoder_last.squeeze(0), q[0].squeeze(0)).cpu().item())
    mmd_image_k_list.append(mmd(image_encoder_last.squeeze(0), k[0].squeeze(0)).cpu().item())
    mmd_image_v_list.append(mmd(image_encoder_last.squeeze(0), v[0].squeeze(0)).cpu().item())
    
    # print(text_encoder_last.squeeze(0).shape)
    # print(q[0].squeeze(0).shape)
    
    cka_text_q_list.append(linear_CKA(text_encoder_last.squeeze(0), q[0].squeeze(0)).cpu().item())
    cka_image_k_list.append(linear_CKA(image_encoder_last.squeeze(0), k[0].squeeze(0)).cpu().item())
    cka_image_v_list.append(linear_CKA(image_encoder_last.squeeze(0), v[0].squeeze(0)).cpu().item())

    if i == 5:
        break

mmd_text_q_avg = np.mean(mmd_text_q_list)
mmd_image_k_avg = np.mean(mmd_image_k_list)
mmd_image_v_avg = np.mean(mmd_image_v_list)

cka_text_q_avg = np.mean(cka_text_q_list)
cka_image_k_avg = np.mean(cka_image_k_list)
cka_image_v_avg = np.mean(cka_image_v_list)

print("\n--- MMD Results ---")
print(f"MMD (Text Encoder vs Q): {mmd_text_q_avg:.4f}")
print(f"MMD (Image Encoder vs K): {mmd_image_k_avg:.4f}")
print(f"MMD (Image Encoder vs V): {mmd_image_v_avg:.4f}")

print(f"CKA (Text Encoder vs Q): {cka_text_q_avg:.4f}")
print(f"CKA (Image Encoder vs K): {cka_image_k_avg:.4f}")
print(f"CKA (Image Encoder vs V): {cka_image_v_avg:.4f}")

from scipy.stats import wasserstein_distance
from torch.nn.functional import cosine_similarity

def wasserstein_dist(X, Y):
    """
    Compute the Wasserstein-1 distance (Earth Mover's Distance) between two sets of embeddings.
    X, Y: tensors of shape (n_samples, n_features)
    """
    return wasserstein_distance(X.cpu().numpy().flatten(), Y.cpu().numpy().flatten())

def cosine_similarity_centroids(X, Y):
    """
    Compute the cosine similarity between the centroids of two distributions.
    """
    X_centroid = X.mean(dim=0, keepdim=True)
    Y_centroid = Y.mean(dim=0, keepdim=True)
    return cosine_similarity(X_centroid, Y_centroid).item()

def norm_diff_centroids(X, Y):
    """
    Compute the norm difference between group centroids.
    """
    X_centroid = X.mean(dim=0)
    Y_centroid = Y.mean(dim=0)
    return torch.norm(X_centroid - Y_centroid, p=2).item()

wasserstein_text_q_list = []
wasserstein_image_k_list = []
wasserstein_image_v_list = []

cosine_text_q_list = []
cosine_image_k_list = []
cosine_image_v_list = []

norm_text_q_list = []
norm_image_k_list = []
norm_image_v_list = []

for i, sample in enumerate(fairface):
    images = transform(sample['image']).unsqueeze(0).to(device)
    captions = f"A {sample['age']} year old {sample['gender']} with {sample['race']} ethnicity."
    
    with torch.no_grad():
        multimodal_features, text_encoder_last, image_encoder_last, q, k, v = model(images, captions, mode="multimodal")
    
    wasserstein_text_q_list.append(wasserstein_dist(text_encoder_last.squeeze(0), q[0].squeeze(0)))
    wasserstein_image_k_list.append(wasserstein_dist(image_encoder_last.squeeze(0), k[0].squeeze(0)))
    wasserstein_image_v_list.append(wasserstein_dist(image_encoder_last.squeeze(0), v[0].squeeze(0)))
    
    cosine_text_q_list.append(cosine_similarity_centroids(text_encoder_last.squeeze(0), q[0].squeeze(0)))
    cosine_image_k_list.append(cosine_similarity_centroids(image_encoder_last.squeeze(0), k[0].squeeze(0)))
    cosine_image_v_list.append(cosine_similarity_centroids(image_encoder_last.squeeze(0), v[0].squeeze(0)))
    
    norm_text_q_list.append(norm_diff_centroids(text_encoder_last.squeeze(0), q[0].squeeze(0)))
    norm_image_k_list.append(norm_diff_centroids(image_encoder_last.squeeze(0), k[0].squeeze(0)))
    norm_image_v_list.append(norm_diff_centroids(image_encoder_last.squeeze(0), v[0].squeeze(0)))
    
    if i == 5:
        break

print("\n--- Wasserstein Distance Results ---")
print(f"Wasserstein (Text Encoder vs Q): {np.mean(wasserstein_text_q_list):.4f}")
print(f"Wasserstein (Image Encoder vs K): {np.mean(wasserstein_image_k_list):.4f}")
print(f"Wasserstein (Image Encoder vs V): {np.mean(wasserstein_image_v_list):.4f}")

print("\n--- Cosine Similarity Between Centroids ---")
print(f"Cosine (Text Encoder vs Q): {np.mean(cosine_text_q_list):.4f}")
print(f"Cosine (Image Encoder vs K): {np.mean(cosine_image_k_list):.4f}")
print(f"Cosine (Image Encoder vs V): {np.mean(cosine_image_v_list):.4f}")

print("\n--- Norm Difference Between Centroids ---")
print(f"Norm Diff (Text Encoder vs Q): {np.mean(norm_text_q_list):.4f}")
print(f"Norm Diff (Image Encoder vs K): {np.mean(norm_image_k_list):.4f}")
print(f"Norm Diff (Image Encoder vs V): {np.mean(norm_image_v_list):.4f}")

def kl_divergence(p, q):
    """ Compute KL divergence between two probability distributions """
    return F.kl_div(torch.log(p + 1e-10), q, reduction="batchmean")  # Avoid log(0)

def jensen_shannon_divergence(X, Y):
    """
    Compute Jensen-Shannon Divergence (JSD) between two sets of feature vectors.
    """
    # Convert feature embeddings into probability distributions
    X_prob = F.softmax(X, dim=-1)
    Y_prob = F.softmax(Y, dim=-1)

    # Compute the mean distribution
    M = 0.5 * (X_prob + Y_prob)

    # Compute JSD
    jsd = 0.5 * kl_divergence(X_prob, M) + 0.5 * kl_divergence(Y_prob, M)
    return jsd

def feature_covariance_shift(X, Y):
    """
    Compute the difference between covariance matrices of two sets of feature vectors.
    """
    # Center the features
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # Compute covariance matrices
    cov_X = torch.mm(X_centered.T, X_centered) / (X.shape[0] - 1)
    cov_Y = torch.mm(Y_centered.T, Y_centered) / (Y.shape[0] - 1)

    # Compute Frobenius norm of the difference
    cov_shift = torch.norm(cov_X - cov_Y, p="fro")

    return cov_shift

# Example usage with existing variables
jsd_text_q = jensen_shannon_divergence(text_encoder_last.squeeze(0), q[0].squeeze(0)).cpu().item()
jsd_image_k = jensen_shannon_divergence(image_encoder_last.squeeze(0), k[0].squeeze(0)).cpu().item()

cov_shift_text_q = feature_covariance_shift(text_encoder_last.squeeze(0), q[0].squeeze(0)).cpu().item()
cov_shift_image_k = feature_covariance_shift(image_encoder_last.squeeze(0), k[0].squeeze(0)).cpu().item()

print("\n--- Jensen-Shannon Divergence (JSD) ---")
print(f"JSD (Text Encoder vs Q): {jsd_text_q:.4f}")
print(f"JSD (Image Encoder vs K): {jsd_image_k:.4f}")

print("\n--- Feature Covariance Shift ---")
print(f"Covariance Shift (Text Encoder vs Q): {cov_shift_text_q:.4f}")
print(f"Covariance Shift (Image Encoder vs K): {cov_shift_image_k:.4f}")

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import pdist, squareform
from geomloss import SamplesLoss  # Ensure you have geomloss installed

# ----------------------
# 1️⃣ Canonical Correlation Analysis (CCA)
# ----------------------
def cca_similarity(x, y, num_components=10):
    """
    Compute CCA similarity by finding correlated components.
    Returns the mean correlation of top components.
    """
    x, y = x.cpu().numpy(), y.cpu().numpy()
    cca = CCA(n_components=min(num_components, x.shape[1], y.shape[1]))
    x_c, y_c = cca.fit_transform(x, y)
    return np.mean([np.corrcoef(x_c[:, i], y_c[:, i])[0, 1] for i in range(x_c.shape[1])])

# ----------------------
# 2️⃣ Projection Weighted CCA (PWCCA)
# ----------------------
def pwcca_similarity(x, y):
    """
    Compute PWCCA similarity.
    """
    x, y = x.cpu().numpy(), y.cpu().numpy()
    cca = CCA(n_components=min(x.shape[1], y.shape[1]))
    x_c, y_c = cca.fit_transform(x, y)
    weights = np.linalg.svd(x_c)[1]  # Singular values as weights
    return np.average([np.corrcoef(x_c[:, i], y_c[:, i])[0, 1] for i in range(x_c.shape[1])], weights=weights)

# ----------------------
# 3️⃣ RV Coefficient (Multivariate Generalization of Pearson)
# ----------------------
def rv_coefficient(x, y):
    """
    Compute RV coefficient, a multivariate generalization of squared correlation.
    """
    x, y = x.cpu().numpy(), y.cpu().numpy()
    x_cov = np.dot(x.T, x)
    y_cov = np.dot(y.T, y)
    numerator = np.trace(np.dot(x_cov, y_cov))
    denominator = np.sqrt(np.trace(np.dot(x_cov, x_cov)) * np.trace(np.dot(y_cov, y_cov)))
    return numerator / (denominator + 1e-10)

# ----------------------
# 4️⃣ Distance Correlation (dCorr)
# ----------------------
def distance_correlation(x, y):
    """
    Compute distance correlation, which captures nonlinear relationships.
    """
    x, y = x.cpu().numpy(), y.cpu().numpy()
    def dist_corr(a, b):
        A = squareform(pdist(a, metric='euclidean'))
        B = squareform(pdist(b, metric='euclidean'))
        A -= A.mean(axis=0)[None, :] + A.mean(axis=1)[:, None] - A.mean()
        B -= B.mean(axis=0)[None, :] + B.mean(axis=1)[:, None] - B.mean()
        return np.sum(A * B) / np.sqrt(np.sum(A**2) * np.sum(B**2))
    return dist_corr(x, y)

# ----------------------
# 5️⃣ Mutual Information (MI)
# ----------------------
def mutual_information(x, y):
    """
    Compute mutual information between two embeddings.
    """
    x, y = x.cpu().numpy(), y.cpu().numpy()
    return mutual_info_score(x.flatten(), y.flatten())

# ----------------------
# 6️⃣ Hilbert-Schmidt Independence Criterion (HSIC)
# ----------------------
def hsic(x, y, sigma=1.0):
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion) using an RBF kernel.
    """
    K_XX, K_YY, K_XY = rbf_kernel(torch.tensor(x), torch.tensor(y), sigma)
    return torch.trace(K_XY) / (x.shape[0] ** 2)

# ----------------------
# 7️⃣ Cross-Covariance Similarity
# ----------------------
def cross_covariance(x, y):
    """
    Compute cross-covariance similarity.
    """
    x, y = x - x.mean(dim=0), y - y.mean(dim=0)
    return torch.norm(torch.mm(x.T, y), p='fro').item()

# ----------------------
# 8️⃣ Whitening Kernel Alignment (WKA)
# ----------------------
def whitening_kernel_alignment(x, y):
    """
    Compute Whitening Kernel Alignment (WKA).
    """
    x, y = x.cpu().numpy(), y.cpu().numpy()
    x_cov, y_cov = np.cov(x.T), np.cov(y.T)
    x_white = np.dot(np.linalg.inv(np.sqrt(x_cov)), x.T).T
    y_white = np.dot(np.linalg.inv(np.sqrt(y_cov)), y.T).T
    return np.trace(np.dot(x_white.T, y_white)) / x.shape[1]

# ------------------------------------------------------
# APPLY METRICS TO YOUR EXISTING VARIABLES
# ------------------------------------------------------
metrics = {
    "CCA_text": cca_similarity(text_encoder_last, q[0]),
    "CCA_imgK": cca_similarity(image_encoder_last, k[0]),
    "CCA_imgV": cca_similarity(image_encoder_last, v[0]),
    
    "PWCCA_text": pwcca_similarity(text_encoder_last, q[0]),
    "PWCCA_imgK": pwcca_similarity(image_encoder_last, k[0]),
    "PWCCA_imgV": pwcca_similarity(image_encoder_last, v[0]),
    
    "RV Coefficient": rv_coefficient(text_encoder_last, q[0]),
    "RV Coefficient": rv_coefficient(image_encoder_last, k[0]),
    "RV Coefficient": rv_coefficient(text_encoder_last, v[0]),
    
    "Distance Correlation": distance_correlation(text_encoder_last, q[0]),
    
    "Mutual Information": mutual_information(text_encoder_last, q[0]),
    
    "HSIC": hsic(text_encoder_last, q[0]),
    
    "Cross-Covariance": cross_covariance(text_encoder_last, q[0]),
    
    "WKA": whitening_kernel_alignment(text_encoder_last, q[0])
}

# Print results
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

