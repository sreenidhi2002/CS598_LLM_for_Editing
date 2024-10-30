from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load the OpenImages dataset from Hugging Face
ds = load_dataset("dalle-mini/open-images")

transform = Compose([
    Resize((512, 512)),  # Resize to your preferred size
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common ImageNet normalization
])

# transform each image in the dataset
def transform_examples(batch):
    batch["image"] = [transform(image.convert("RGB")) for image in batch["image"]]
    return batch

# transform the dataset
ds.set_transform(transform_examples)

# batch using dataloader
dataloader = DataLoader(ds["train"], batch_size=8, shuffle=True)

for batch in dataloader:
    images = batch["image"]
    labels = batch.get("label", None)
    # training code
