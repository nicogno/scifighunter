import argparse
import os
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm
import torch
import clip
from PIL import Image
import numpy as np
import pickle
import json

def extract_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem
    fig_dir = Path(output_dir) / pdf_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    img_count = 0

    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        text_blocks = page.get_text("dict")["blocks"]

        for img_index, img_info in enumerate(images): # Renamed 'img' to 'img_info'
            try:
                xref = img_info[0]

                # Get the bounding box of the image on the page
                try:
                    # img_info is the item from page.get_images() list
                    bbox = page.get_image_bbox(img_info)
                except ValueError as e_bbox:
                    print(f"[WARNING] Could not get bbox for image on page {i+1}, img_index {img_index+1} (xref: {xref}). Error: {e_bbox}")
                    continue

                # Validate the obtained bbox
                if not isinstance(bbox, fitz.Rect) or bbox.is_empty or bbox.is_infinite:
                    print(f"[WARNING] Skipping image on page {i+1}, img_index {img_index+1} (xref: {xref}) due to invalid or empty bbox: {bbox}")
                    continue

                # Extract and save image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img_filename = f"page_{i+1}_img_{img_index+1}.{image_ext}"
                img_path = fig_dir / img_filename

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # Find closest text block below the image (possible caption)
                caption_text = ""
                min_dist = float("inf")
                for block in text_blocks:
                    if "lines" not in block or not block["lines"]:
                        continue
                    block_rect = fitz.Rect(block["bbox"])
                    if block_rect.y0 > bbox.y1:  # Below the image
                        dist = block_rect.y0 - bbox.y1
                        if dist < min_dist:
                            min_dist = dist
                            caption_text = " ".join(
                                span["text"]
                                for line in block["lines"]
                                for span in line["spans"]
                            ).strip() # Added strip() here for cleaner captions

                metadata.append({
                    "image_path": str(img_path.relative_to(fig_dir)),
                    "caption": caption_text
                })
                img_count += 1

            except Exception as e:
                print(f"[ERROR] Failed to process image on page {i+1}, img_index {img_index+1} (xref: {img_info[0] if img_info else 'unknown'}): {e}")
                continue

    # Save metadata
    with open(fig_dir / "captions.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return img_count




def extract_from_directory(pdf_dir, output_dir):
    pdf_paths = list(Path(pdf_dir).rglob("*.pdf"))
    total_imgs = 0
    for pdf_path in tqdm(pdf_paths, desc="Extracting figures"):
        total_imgs += extract_images_from_pdf(pdf_path, output_dir)
    print(f"Extracted {total_imgs} images from {len(pdf_paths)} PDFs.")


def build_image_embeddings(figures_dir, output_index):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_embeddings = []
    path_list = []

    for pdf_dir in tqdm(list(Path(figures_dir).iterdir()), desc="Indexing PDFs"):
        captions_file = pdf_dir / "captions.json"
        if not captions_file.exists():
            continue

        with open(captions_file, "r") as f:
            entries = json.load(f)

        for entry in entries:
            try:
                img_path = pdf_dir / entry["image_path"]
                caption = entry.get("caption", "")
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    if caption:
                        # Pre-truncate the caption string to avoid issues with very long captions
                        # CLIP's context length is 77 tokens.
                        # A common heuristic is ~4 chars/token, so 77*4 = 308.
                        # Let's use a slightly conservative character limit.
                        max_caption_char_len = 280  # Max characters for a caption
                        if len(caption) > max_caption_char_len:
                            caption = caption[:max_caption_char_len]
                            # You could add a log here if you want to know when captions are truncated
                            # print(f"[INFO] Truncated caption for {img_path} starting with: {caption[:30]}...")

                        # Explicitly set truncate=True
                        text_tokens = clip.tokenize([caption], truncate=True).to(device)
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        alpha = 1.0  # Weight for image features
                        combined = (alpha * image_features + (1 - alpha) * text_features)
                        combined /= combined.norm(dim=-1, keepdim=True)  # Re-normalize
                    else:
                        combined = image_features # image_features is already normalized

                image_embeddings.append(combined.cpu().numpy())
                path_list.append(str(img_path))
            except Exception as e:
                print(f"Failed processing {entry['image_path']}: {e}")

    embeddings_array = np.vstack(image_embeddings)
    with open(output_index, "wb") as f:
        pickle.dump({"embeddings": embeddings_array, "paths": path_list}, f)

    print(f"Saved {len(path_list)} joint image+caption embeddings to {output_index}")



def search_images(query, index_path, top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    with open(index_path, "rb") as f:
        index = pickle.load(f)

    embeddings = index["embeddings"]
    paths = index["paths"]

    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    similarities = (embeddings @ text_features.T).squeeze()
    top_indices = similarities.argsort()[::-1][:top_k]

    print(f"Top {top_k} matches for: '{query}'")
    for i in top_indices:
        print(f"{paths[i]} (score: {similarities[i]:.4f})")


def main():
    parser = argparse.ArgumentParser(description="FigureFinder: extract and search figures from scientific PDFs")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="Extract figures from PDFs")
    extract_parser.add_argument("--pdf_dir", type=str, required=True)
    extract_parser.add_argument("--output_dir", type=str, required=True)

    embed_parser = subparsers.add_parser("embed", help="Build CLIP embeddings for images")
    embed_parser.add_argument("--figures_dir", type=str, required=True)
    embed_parser.add_argument("--index_file", type=str, required=True)

    search_parser = subparsers.add_parser("search", help="Search for figures using a prompt")
    search_parser.add_argument("--index_file", type=str, required=True)
    search_parser.add_argument("--query", type=str, required=True)
    search_parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "extract":
        extract_from_directory(args.pdf_dir, args.output_dir)
    elif args.command == "embed":
        build_image_embeddings(args.figures_dir, args.index_file)
    elif args.command == "search":
        search_images(args.query, args.index_file, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

