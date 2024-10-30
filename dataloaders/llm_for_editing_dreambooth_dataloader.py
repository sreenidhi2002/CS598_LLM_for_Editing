import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import load_dataset
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# link https://huggingface.co/datasets/google/dreambooth

# List of DreamBooth categories to load
categories = ["backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can", "candle", "cat", "cat2", "clock", "colorful_sneaker", "default", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8", "duck_toy", "fancy_boot", "grey_sloth_plushie", "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon", "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie"]

# transformations
transform = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# function to apply transformations
def transform_examples(batch):
    batch["image"] = [transform(image.convert("RGB")) for image in batch["image"]]
    return batch

# Load and transform each category dataset
datasets = []
for category in categories:
    ds = load_dataset("google/dreambooth", category)
    ds.set_transform(transform_examples)
    datasets.append(ds["train"])


combined_dataset = ConcatDataset(datasets)

# DataLoader for the combined dataset
dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True)

# iterating through the DataLoader
for batch in dataloader:
    images = batch["image"]
    labels = batch.get("label", None)
    # Training code
