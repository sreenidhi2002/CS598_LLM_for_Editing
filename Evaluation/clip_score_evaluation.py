import torch
import clip
from PIL import Image
import os
import argparse


def calculate_clip_score(image_path, text, model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    text_tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).item()

    return similarity


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_paths = []
    texts = []
    with open(args.input_file, "r") as f:
        for line in f:
            image_path, text = line.strip().split("\t")
            image_paths.append(image_path)
            texts.append(text)

    clip_scores = []
    for image_path, text in zip(image_paths, texts):
        if not os.path.exists(image_path):
            print(f"Warning: Image '{image_path}' not found. Skipping.")
            continue
        score = calculate_clip_score(image_path, text, model, preprocess, device)
        clip_scores.append((image_path, text, score))

    with open(args.output_file, "w") as f:
        for image_path, text, score in clip_scores:
            f.write(f"{image_path}\t{text}\t{score:.4f}\n")

    print(f"Evaluation completed. Results saved to '{args.output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate CLIP scores for image-text pairs."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file containing image paths and text pairs (tab-separated).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file where CLIP scores will be saved.",
    )
    args = parser.parse_args()
    main(args)
