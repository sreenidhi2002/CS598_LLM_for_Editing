import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Load a pre-trained vision model (e.g., ResNet50)
model = models.resnet50(pretrained=True)
model = model.eval()  # Set to evaluation mode

# Remove the classification layer to get the embedding
model = torch.nn.Sequential(*list(model.children())[:-1])

# Check if MPS is available for Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

# Define image transformations to match the model's expected input
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract embeddings from an image
def extract_embedding(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get the embedding
    with torch.no_grad():
        embedding = model(input_tensor)
    embedding = embedding.squeeze().cpu().numpy()

    return embedding

# Example usage
image_path = "path/to/your/image.jpg"
embedding = extract_embedding(image_path)
print("Embedding shape:", embedding.shape)  # Should be (2048,) for ResNet50
print("Embedding:", embedding)
