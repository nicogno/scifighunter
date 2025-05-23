import argparse
import os
from pathlib import Path
import json
import shutil
import tempfile
import time
import re
import pkg_resources
import torch
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import open_clip
import chromadb
from scifighunter_advanced_filters import ImageClassifier, PostRetrievalFilter

# Constants
OPENCLIP_MODEL_NAME = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"
CHROMA_COLLECTION_NAME = "scientific_figures"
PYMUPDF_MIN_VERSION = "1.18.0"
PYMUPDF_MAX_VERSION = "1.22.0"

def check_pymupdf_version():
    """Check if the installed PyMuPDF version is compatible.
    
    Returns:
        tuple: (is_compatible, message)
    """
    try:
        version = pkg_resources.get_distribution("PyMuPDF").version
        min_version = pkg_resources.parse_version(PYMUPDF_MIN_VERSION)
        max_version = pkg_resources.parse_version(PYMUPDF_MAX_VERSION)
        current_version = pkg_resources.parse_version(version)
        
        if current_version < min_version:
            return False, f"PyMuPDF version {version} is older than recommended minimum {PYMUPDF_MIN_VERSION}. This may cause extraction issues."
        elif current_version > max_version:
            return False, f"PyMuPDF version {version} is newer than recommended maximum {PYMUPDF_MAX_VERSION}. This may cause extraction issues."
        else:
            return True, f"PyMuPDF version {version} is compatible."
    except Exception as e:
        return False, f"Error checking PyMuPDF version: {e}"

def extract_from_pdf(pdf_path, output_dir, use_classifier=False):
    """Extract images from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        use_classifier: Whether to use image classification to filter out non-scientific images
        
    Returns:
        list: Metadata for extracted images
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier if requested
    classifier = None
    if use_classifier:
        classifier = ImageClassifier()
    
    # Extract PDF filename without extension
    pdf_name = pdf_path.stem
    
    # Create a subdirectory for this PDF
    pdf_output_dir = output_dir / pdf_name
    pdf_output_dir.mkdir(exist_ok=True)
    
    # Open the PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []
    
    # Extract images from each page
    extracted_metadata = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        
        # Get page text for caption extraction
        page_text = page.get_text()
        
        # Extract images
        image_list = page.get_images(full=True)
        
        for img_idx, img_info in enumerate(image_list):
            try:
                # Get image data
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Skip small images (likely icons, bullets, etc.)
                if len(image_bytes) < 100:  # Very small image
                    continue
                
                # Save the image
                image_filename = f"{pdf_name}_page{page_idx+1}_img{img_idx+1}.{image_ext}"
                image_path = pdf_output_dir / image_filename
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Get image dimensions
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception:
                    width, height = 0, 0
                
                # Skip very small images
                if width < 100 or height < 100:
                    os.remove(image_path)
                    continue
                
                # Skip images with extreme aspect ratios
                if width > 0 and height > 0:
                    aspect_ratio = width / height
                    if aspect_ratio < 0.2 or aspect_ratio > 5:
                        os.remove(image_path)
                        continue
                
                # Apply image classification if requested
                if classifier:
                    is_scientific, confidence, category = classifier.is_scientific_figure(image_path)
                    if not is_scientific:
                        print(f"Filtered out non-scientific image: {image_path} (confidence: {confidence:.2f}, category: {category})")
                        os.remove(image_path)
                        continue
                
                # Extract caption (simple heuristic: text near the image)
                # This is a simplified approach; a more sophisticated method would be better
                caption = extract_caption_for_image(page_text, img_idx)
                
                # Filter out images with captions containing certain keywords
                if should_filter_by_caption(caption):
                    os.remove(image_path)
                    continue
                
                # Add metadata
                metadata = {
                    "pdf_name": pdf_name,
                    "page_number": page_idx + 1,
                    "image_index": img_idx + 1,
                    "image_path": str(image_path),
                    "caption": caption,
                    "width": width,
                    "height": height
                }
                
                extracted_metadata.append(metadata)
                
            except Exception as e:
                print(f"Error extracting image {img_idx} from page {page_idx+1} of {pdf_path}: {e}")
    
    return extracted_metadata

def extract_from_directory(pdf_dir, output_dir, use_classifier=False):
    """Extract images from all PDFs in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save extracted images
        use_classifier: Whether to use image classification to filter out non-scientific images
        
    Returns:
        list: Metadata for all extracted images
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    
    # Get all PDF files in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []
    
    # Extract images from each PDF
    all_metadata = []
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        metadata = extract_from_pdf(pdf_file, output_dir, use_classifier)
        all_metadata.extend(metadata)
        print(f"Extracted {len(metadata)} images from {pdf_file}")
    
    return all_metadata

def extract_caption_for_image(page_text, img_idx):
    """Extract a caption for an image based on page text.
    
    Args:
        page_text: Text content of the page
        img_idx: Index of the image on the page
        
    Returns:
        str: Extracted caption
    """
    # This is a simplified approach; TODO: improve caption extraction
    
    # Look for figure captions
    caption_patterns = [
        r"(?:Figure|Fig\.?)\s+\d+[.:]\s*([^\n]+)",
        r"(?:Table)\s+\d+[.:]\s*([^\n]+)",
        r"(?:Scheme|Chart|Plate)\s+\d+[.:]\s*([^\n]+)"
    ]
    
    for pattern in caption_patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        if matches and len(matches) > img_idx:
            return matches[img_idx]
    
    # If no specific caption found, return empty string
    return ""

def should_filter_by_caption(caption):
    """Check if an image should be filtered based on its caption.
    
    Args:
        caption: Image caption
        
    Returns:
        bool: True if the image should be filtered out
    """
    if not caption:
        return False
    
    # Keywords that indicate non-scientific images
    filter_keywords = [
        "copyright", "©", "all rights reserved", "trademark", "logo",
        "journal", "publisher", "cover", "license"
    ]
    
    caption_lower = caption.lower()
    
    for keyword in filter_keywords:
        if keyword in caption_lower:
            return True
    
    return False

def build_image_embeddings(output_dir, metadata_list):
    """Build embeddings for extracted images using OpenCLIP.
    
    Args:
        output_dir: Directory containing extracted images
        metadata_list: List of metadata for extracted images
        
    Returns:
        bool: True if successful
    """
    output_dir = Path(output_dir)
    
    # Create chroma_db directory if it doesn't exist
    chroma_db_dir = output_dir / "chroma_db"
    chroma_db_dir.mkdir(parents=True, exist_ok=True)
    
    # Load OpenCLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL_NAME,
        pretrained=OPENCLIP_PRETRAINED,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_NAME)
    
    # Initialize ChromaDB collection
    client = chromadb.PersistentClient(path=str(chroma_db_dir))
    
    # Delete collection if it exists
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except:
        pass
    
    # Create collection
    collection = client.create_collection(CHROMA_COLLECTION_NAME)
    
    # Process images in batches
    batch_size = 10
    for i in range(0, len(metadata_list), batch_size):
        batch = metadata_list[i:i+batch_size]
        
        # Prepare batch data
        ids = []
        embeddings = []
        metadatas = []
        
        for metadata in batch:
            try:
                image_path = metadata["image_path"]
                caption = metadata["caption"]
                
                # Generate unique ID
                image_id = f"{metadata['pdf_name']}_{metadata['page_number']}_{metadata['image_index']}"
                
                # Load and preprocess image
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Generate image embedding
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_embedding = image_features.cpu().numpy().flatten().tolist()
                
                # Generate text embedding for caption
                if caption:
                    with torch.no_grad():
                        text_tokens = tokenizer([caption]).to(device)
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        text_embedding = text_features.cpu().numpy().flatten().tolist()
                else:
                    # If no caption, use a zero vector
                    text_embedding = [0.0] * len(image_embedding)
                
                # Combine image and text embeddings (50/50 weight)
                combined_embedding = [(i + t) / 2 for i, t in zip(image_embedding, text_embedding)]
                
                ids.append(image_id)
                embeddings.append(combined_embedding)
                metadatas.append(metadata)
                
            except Exception as e:
                print(f"Error processing {metadata.get('image_path', 'unknown')}: {e}")
        
        # Add batch to collection
        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
    
    return True

def search_images(query, top_k=5, use_post_filtering=True, aggressive_filtering=True):
    """Search for images similar to a text query.
    
    Args:
        query: Text query
        top_k: Number of results to return
        use_post_filtering: Whether to apply post-retrieval filtering
        aggressive_filtering: Whether to use aggressive filtering to remove problematic images
        
    Returns:
        dict: Search results
    """
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL_NAME,
        pretrained=OPENCLIP_PRETRAINED,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_NAME)
    
    # Get ChromaDB collection
    collection = get_chroma_collection()
    
    # Convert query to embedding
    with torch.no_grad():
        text_tokens = tokenizer([query]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_embedding = text_features.cpu().numpy().flatten().tolist()
    
    # Search for similar embeddings in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,  # Get more results than needed for filtering
        include=["metadatas", "distances"]
    )
    
    # Apply post-retrieval filtering if requested
    if use_post_filtering:
        post_filter = PostRetrievalFilter(aggressive_mode=aggressive_filtering)
        results = post_filter.filter_results(results, query, max_results=top_k)
    
    return results

def get_chroma_collection():
    """Get the ChromaDB collection for scientific figures.
    
    Returns:
        Collection: ChromaDB collection
    """
    # Find the chroma_db directory
    # Start with the current directory and look for chroma_db
    current_dir = Path.cwd()
    chroma_db_dir = None
    
    # Check current directory and its subdirectories
    for path in current_dir.glob("**/chroma_db"):
        if path.is_dir():
            chroma_db_dir = path
            break
    
    # If not found, check for output directory and create chroma_db there
    if not chroma_db_dir:
        # Look for output directory
        output_dir = None
        for path in current_dir.glob("**/output"):
            if path.is_dir():
                output_dir = path
                break
        
        # If output directory found, create chroma_db there
        if output_dir:
            chroma_db_dir = output_dir / "chroma_db"
            chroma_db_dir.mkdir(parents=True, exist_ok=True)
        else:
            # If no output directory, create in current directory
            chroma_db_dir = current_dir / "output" / "chroma_db"
            chroma_db_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(chroma_db_dir))
    
    # Get collection or create if it doesn't exist
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"Collection not found, creating new collection: {e}")
        collection = client.create_collection(CHROMA_COLLECTION_NAME)
    
    return collection

def reindex_database(pdf_dir, output_dir, use_classifier=True):
    """Re-extract and re-index all PDFs with the classifier enabled.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save extracted images
        use_classifier: Whether to use image classification to filter out non-scientific images
        
    Returns:
        bool: True if successful
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    
    # Create a backup of the original output directory
    backup_dir = output_dir.parent / f"{output_dir.name}_backup_{int(time.time())}"
    if output_dir.exists():
        shutil.copytree(output_dir, backup_dir)
        print(f"Created backup of original data at {backup_dir}")
    
    # Clear the output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract images from PDFs
    print(f"Re-extracting images from PDFs in {pdf_dir}...")
    all_metadata = extract_from_directory(pdf_dir, output_dir, use_classifier=use_classifier)
    
    if not all_metadata:
        print("No images extracted.")
        return False
    
    # Build embeddings
    print(f"Building embeddings for {len(all_metadata)} images...")
    success = build_image_embeddings(output_dir, all_metadata)
    
    if success:
        print("Database re-indexing completed successfully.")
        return True
    else:
        print("Database re-indexing failed.")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SciFigHunter: Scientific Figure Search")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract images from PDFs")
    extract_parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    extract_parser.add_argument("--output_dir", required=True, help="Directory to save extracted images")
    extract_parser.add_argument("--use_classifier", action="store_true", help="Use image classification to filter out non-scientific images")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for images")
    search_parser.add_argument("--query", required=True, help="Text query")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--output_dir", default=".", help="Directory containing the chroma_db")
    search_parser.add_argument("--use_post_filtering", action="store_true", help="Apply post-retrieval filtering")
    search_parser.add_argument("--aggressive_filtering", action="store_true", help="Use aggressive filtering to remove problematic images")
    
    # Reindex command
    reindex_parser = subparsers.add_parser("reindex", help="Re-extract and re-index all PDFs")
    reindex_parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    reindex_parser.add_argument("--output_dir", required=True, help="Directory to save extracted images")
    reindex_parser.add_argument("--use_classifier", action="store_true", help="Use image classification to filter out non-scientific images")
    
    args = parser.parse_args()
    
    # Check PyMuPDF version
    is_compatible, version_message = check_pymupdf_version()
    if not is_compatible:
        print(f"[WARNING] {version_message}")
    
    if args.command == "extract":
        # Extract images from PDFs
        all_metadata = extract_from_directory(args.pdf_dir, args.output_dir, args.use_classifier)
        
        if all_metadata:
            # Build embeddings
            build_image_embeddings(args.output_dir, all_metadata)
            print(f"Extracted and embedded {len(all_metadata)} images.")
        else:
            print("No images extracted.")
    
    elif args.command == "search":
        # Search for images
        try:
            results = search_images(args.query, args.top_k, args.use_post_filtering, args.aggressive_filtering)
            
            # Print results
            if results and results["ids"] and results["ids"][0]:
                print(f"Found {len(results['ids'][0])} results:")
                for i in range(len(results["ids"][0])):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    score = 1 - distance
                    
                    print(f"\nResult {i+1}:")
                    print(f"  PDF: {metadata.get('pdf_name', 'unknown')}, Page: {metadata.get('page_number', 'unknown')}")
                    print(f"  Score: {score:.4f}")
                    print(f"  Caption: {metadata.get('caption', '')[:100]}...")
                    print(f"  Image path: {metadata.get('image_path', 'unknown')}")
            else:
                print("No results found.")
        
        except Exception as e:
            print(f"Error during search: {e}")
    
    elif args.command == "reindex":
        # Re-extract and re-index all PDFs
        reindex_database(args.pdf_dir, args.output_dir, args.use_classifier)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
