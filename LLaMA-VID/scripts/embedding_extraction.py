import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as transforms

model_name = "path_to_your_pretrained_model"
model = AutoModel.from_pretrained(model_name)


image_path = "path_to_image.jpg"
image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose(
    [
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image_tensor = preprocess(image).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_tensor = image_tensor.to(device)

with torch.no_grad():
    vis_embed = model.visual_encoder(image_tensor)

print("Image Embedding Shape:", vis_embed.shape)
