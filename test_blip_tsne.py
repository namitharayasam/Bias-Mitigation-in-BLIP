from datasets import load_dataset
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from models.blip import BLIP_Base, load_checkpoint

# Load dataset
# fairface = load_dataset("HuggingFaceM4/FairFace", "0.25", split="train", trust_remote_code=True)
# print(fairface[4])
# fairface = load_dataset("HuggingFaceM4/m4-bias-eval-fair-face", split="train")
fairface = load_dataset("HuggingFaceM4/m4-bias-eval-stable-bias", split="train")

fairface_subset = fairface.select(range(300))

# Preprocess images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def preprocess_sample(sample):
    image = sample['image']
    image = image.convert("RGB")  
    image = transform(image)
    caption = "Describe the image."
    return image, caption

def extract_embeddings(model, dataset, batch_size=8):
    image_embeds_list, text_embeds_list, multimodal_embeds_list = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))  # ✅ Use .select()

        images, captions = zip(*[preprocess_sample(sample) for sample in batch])

        images = torch.stack(images).to(device)
        captions = list(captions)

        # Extract embeddings
        with torch.no_grad():
            image_embeds = model(images, captions, mode="image")
            text_embeds = model(images, captions, mode="text")
            multimodal_embeds = model(images, captions, mode="multimodal")

        image_embeds_list.append(image_embeds)
        text_embeds_list.append(text_embeds)
        multimodal_embeds_list.append(multimodal_embeds)

    # Concatenate all batches into one large tensor
    image_embeds_all = torch.cat(image_embeds_list, dim=0)
    text_embeds_all = torch.cat(text_embeds_list, dim=0)
    multimodal_embeds_all = torch.cat(multimodal_embeds_list, dim=0)

    return image_embeds_all, text_embeds_all, multimodal_embeds_all

def visualize_tsne(image_embeds, text_embeds, multimodal_embeds):
    image_embeds = image_embeds.detach().cpu().numpy().mean(axis=1)
    text_embeds = text_embeds.detach().cpu().numpy().mean(axis=1)
    multimodal_embeds = multimodal_embeds.detach().cpu().numpy().mean(axis=1)

    all_embeds = np.concatenate([image_embeds, text_embeds, multimodal_embeds])
    n_samples = all_embeds.shape[0]
    perplexity_value = min(30, n_samples - 1)  # Adjust perplexity based on dataset size
    
    print(f"Running t-SNE with perplexity={perplexity_value} (n_samples={n_samples})...")
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    tsne_results = tsne.fit_transform(all_embeds)
    
    plt.figure(figsize=(10, 7))
    num_samples = image_embeds.shape[0]
    plt.scatter(tsne_results[:num_samples, 0], tsne_results[:num_samples, 1], label="Image Embeddings", alpha=0.7, c="blue")
    plt.scatter(tsne_results[num_samples:num_samples * 2, 0], tsne_results[num_samples:num_samples * 2, 1], label="Text Embeddings", alpha=0.7, c="green")
    plt.scatter(tsne_results[num_samples * 2:, 0], tsne_results[num_samples * 2:, 1], label="Multimodal Embeddings", alpha=0.7, c="red")
    
    plt.ylim(-30, 60)
    plt.legend()
    plt.title("t-SNE Visualization of Dataset Embeddings")
    plt.show()

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BLIP_Base(image_size=224, vit="base")
model, msg = load_checkpoint(model, "checkpoints/model_base_14M.pth")
model = model.to(device)
model.eval()

# Extract embeddings for the dataset
image_embeds, text_embeds, multimodal_embeds = extract_embeddings(model, fairface_subset, batch_size=8)

# Run t-SNE Visualization
visualize_tsne(image_embeds, text_embeds, multimodal_embeds)