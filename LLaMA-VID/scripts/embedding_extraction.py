import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTConfig
from tqdm import tqdm

# Set the device to CPU (or change to "mps" if available on an M1 Mac)
device = torch.device("mps") if torch.has_mps else torch.device("cpu")

# Paths
# link for model weights: https://drive.google.com/file/d/1TkYFParUTx_yW-HPvSINFVCDsU9ZrLsR/view?usp=drive_link
model_path = "../model_zoo/LAVIS/eva_vit_g.pth"
image_path = "../../outputs/blip-backpack_on_the_floor.png"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((336, 336)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Load a ViT model configuration similar to EVA-G
config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, image_size=336, patch_size=16)
model = ViTModel(config).to(device)

# Load the EVA-G weights
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore mismatches

# Extract embeddings
with torch.no_grad():
    print("Extracting image embeddings using EVA-G...")
    outputs = model(pixel_values=image_tensor)
    embeddings = outputs.last_hidden_state

# Display the shape and content of the embeddings
print("Image Embeddings Shape:", embeddings.shape)
print("Image Embeddings:", embeddings)