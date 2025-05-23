import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from PIL import Image
import re
from pathlib import Path

class ImageClassifier:
    """Lightweight classifier for scientific figures."""
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.categories = ["SCIENTIFIC", "NON-SCIENTIFIC"]
    
    def _load_model(self):
        """Load the model on demand to save resources."""
        if self.model is None:
            # Use MobileNetV2 for efficiency
            self.model = torchvision.models.mobilenet_v2(pretrained=True)
            # Modify the classifier for binary classification
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
            self.model.to(self.device)
            self.model.eval()
            
            # For now, we'll use a pretrained model and rely on simple heuristics
            # In a production system, you would fine-tune this on scientific figures
    
    def is_scientific_figure(self, image_path):
        """Determine if an image is a scientific figure.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (is_scientific, confidence, category)
        """
        # Load the model if not already loaded
        self._load_model()
        
        # Simple heuristics for common non-scientific images
        if self._is_simple_shape(image_path):
            return False, 0.95, "SIMPLE_SHAPE"
        
        # For now, use a simple heuristic approach
        # In a production system, you would use the actual model prediction
        
        # Open the image
        try:
            img = Image.open(image_path).convert("RGB")
            
            # Check if it's a solid color or very simple image
            img_np = np.array(img)
            if self._is_mostly_solid_color(img_np):
                return False, 0.90, "SOLID_COLOR"
            
            # Check if it's a logo or icon (typically small with few colors)
            width, height = img.size
            if width < 200 and height < 200:
                unique_colors = len(np.unique(img_np.reshape(-1, 3), axis=0))
                if unique_colors < 20:  # Very few colors
                    return False, 0.85, "LOGO_ICON"
            
            # For demonstration, we'll use a simple rule:
            # Images with more complexity are more likely to be scientific
            edges = cv2.Canny(img_np, 100, 200)
            edge_density = np.count_nonzero(edges) / (width * height)
            
            if edge_density < 0.01:  # Very few edges
                return False, 0.80, "LOW_COMPLEXITY"
            elif edge_density > 0.2:  # Very high edge density (like text or complex diagrams)
                return True, 0.75, "HIGH_COMPLEXITY"
            
            # Default to scientific for now
            return True, 0.60, "DEFAULT"
            
        except Exception as e:
            print(f"Error classifying image {image_path}: {e}")
            return True, 0.50, "ERROR"  # Default to keeping the image if there's an error
    
    def _is_simple_shape(self, image_path):
        """Check if the image is a simple geometric shape."""
        try:
            # Read the image
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours or too many, it's not a simple shape
            if len(contours) == 0 or len(contours) > 5:
                return False
            
            # Check if the largest contour is a circle or rectangle
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if it's a circle
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.8:  # Close to a circle
                    return True
            
            # Check if it's a rectangle
            approx = cv2.approxPolyDP(largest_contour, 0.04 * perimeter, True)
            if len(approx) == 4:  # Rectangle has 4 vertices
                return True
            
            return False
        
        except Exception as e:
            print(f"Error checking if image is a simple shape: {e}")
            return False
    
    def _is_mostly_solid_color(self, img_np):
        """Check if the image is mostly a solid color."""
        try:
            # Resize for efficiency
            small = cv2.resize(img_np, (50, 50))
            
            # Calculate color variance
            variance = np.var(small, axis=(0, 1)).sum()
            
            # If variance is very low, it's mostly a solid color
            return variance < 500
        
        except Exception as e:
            print(f"Error checking if image is mostly solid color: {e}")
            return False


class PostRetrievalFilter:
    """Filter for post-retrieval processing of search results."""
    
    def __init__(self, aggressive_mode=False):
        """Initialize the filter.
        
        Args:
            aggressive_mode: Whether to use aggressive filtering to completely remove problematic images
        """
        self.aggressive_mode = aggressive_mode
    
    def filter_results(self, results, query, max_results=10):
        """Filter and re-rank search results.
        
        Args:
            results: Search results from ChromaDB
            query: Original search query
            max_results: Maximum number of results to return
            
        Returns:
            dict: Filtered and re-ranked results
        """
        if not results or not results["ids"] or not results["ids"][0]:
            return results
        
        # Make a copy of the results
        filtered_results = {
            "ids": [results["ids"][0].copy()],
            "metadatas": [results["metadatas"][0].copy()],
            "distances": [results["distances"][0].copy()]
        }
        
        # Apply aggressive filtering if enabled
        if self.aggressive_mode:
            # Filter out problematic images
            indices_to_keep = []
            
            for i in range(len(filtered_results["ids"][0])):
                metadata = filtered_results["metadatas"][0][i]
                
                # Skip circular images
                image_path = metadata.get("image_path", "")
                if image_path and self._is_circular_image(image_path):
                    print(f"Filtering out circular image: {image_path}")
                    continue
                
                # Skip publisher logos
                if self._is_publisher_logo(metadata):
                    print(f"Filtering out publisher logo: {image_path}")
                    continue
                
                # Skip images with mostly solid color
                if image_path and os.path.exists(image_path):
                    try:
                        img = cv2.imread(image_path)
                        if img is not None and self._is_mostly_solid_color(img):
                            print(f"Filtering out solid color image: {image_path}")
                            continue
                    except Exception as e:
                        print(f"Error checking solid color: {e}")
                
                indices_to_keep.append(i)
            
            # Update the results with only the kept indices
            if indices_to_keep:
                filtered_results["ids"][0] = [filtered_results["ids"][0][i] for i in indices_to_keep]
                filtered_results["metadatas"][0] = [filtered_results["metadatas"][0][i] for i in indices_to_keep]
                filtered_results["distances"][0] = [filtered_results["distances"][0][i] for i in indices_to_keep]
            else:
                # If all results were filtered out, return empty results
                return {"ids": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Apply re-ranking
        scores = []
        
        for i in range(len(filtered_results["ids"][0])):
            metadata = filtered_results["metadatas"][0][i]
            distance = filtered_results["distances"][0][i]
            
            # Base score is inverse of distance
            score = 1 - distance
            
            # Boost scientific figures
            caption = metadata.get("caption", "").lower()
            if self._is_scientific_caption(caption):
                score += 0.5
            
            # Boost results with query terms in caption
            if self._caption_contains_query_terms(caption, query):
                score += 0.3
            
            # Penalize first page results (often contain logos/headers)
            if metadata.get("page_number", 0) == 1:
                score -= 0.2
            
            scores.append(score)
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        
        # Limit to max_results
        sorted_indices = sorted_indices[:max_results]
        
        # Create new results
        new_results = {
            "ids": [[filtered_results["ids"][0][i] for i in sorted_indices]],
            "metadatas": [[filtered_results["metadatas"][0][i] for i in sorted_indices]],
            "distances": [[filtered_results["distances"][0][i] for i in sorted_indices]]
        }
        
        return new_results
    
    def _is_circular_image(self, image_path):
        """Check if the image is circular."""
        try:
            # Read the image
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours, it's not a circle
            if len(contours) == 0:
                return False
            
            # Check if the largest contour is a circle
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check circularity
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.85:  # Very close to a circle
                    # Also check if it's mostly a solid color
                    if self._is_mostly_solid_color(img):
                        return True
            
            return False
        
        except Exception as e:
            print(f"Error checking if image is circular: {e}")
            return False
    
    def _is_mostly_solid_color(self, img):
        """Check if the image is mostly a solid color."""
        try:
            # Resize for efficiency
            small = cv2.resize(img, (50, 50))
            
            # Calculate color variance
            variance = np.var(small, axis=(0, 1)).sum()
            
            # If variance is very low, it's mostly a solid color
            return variance < 500
        
        except Exception as e:
            print(f"Error checking if image is mostly solid color: {e}")
            return False
    
    def _is_publisher_logo(self, metadata):
        """Check if the image is likely a publisher logo."""
        # Check caption for publisher-related keywords
        caption = metadata.get("caption", "").lower()
        publisher_keywords = [
            "elsevier", "springer", "wiley", "nature", "science", "cell press",
            "copyright", "©", "all rights reserved", "trademark", "logo",
            "journal", "publisher", "license"
        ]
        
        for keyword in publisher_keywords:
            if keyword in caption:
                return True
        
        # Check if it's on the first page and has a generic caption
        if metadata.get("page_number", 0) == 1:
            generic_captions = [
                "cover", "title", "header", "footer", "published", "journal",
                "volume", "issue", "doi", "copyright", "license"
            ]
            
            for term in generic_captions:
                if term in caption:
                    return True
        
        return False
    
    def _is_scientific_caption(self, caption):
        """Check if the caption is scientific."""
        if not caption:
            return False
        
        scientific_terms = [
            "figure", "fig", "table", "chart", "graph", "plot", "diagram",
            "image", "illustration", "map", "photo", "picture", "scan",
            "mri", "ct", "pet", "fmri", "microscopy", "histology",
            "analysis", "result", "data", "experiment", "study", "research",
            "patient", "subject", "sample", "specimen", "cell", "tissue",
            "brain", "tumor", "cancer", "disease", "treatment", "therapy"
        ]
        
        for term in scientific_terms:
            if re.search(r'\b' + term + r'\b', caption, re.IGNORECASE):
                return True
        
        return False
    
    def _caption_contains_query_terms(self, caption, query):
        """Check if the caption contains terms from the query."""
        if not caption or not query:
            return False
        
        # Tokenize query and caption
        query_terms = query.lower().split()
        caption_lower = caption.lower()
        
        # Check if any query term is in the caption
        for term in query_terms:
            if len(term) > 2 and re.search(r'\b' + re.escape(term) + r'\b', caption_lower):
                return True
        
        return False


# For testing
if __name__ == "__main__":
    import os
    
    # Test the classifier
    classifier = ImageClassifier()
    
    # Test images
    test_images = [
        "test_image1.jpg",
        "test_image2.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            is_scientific, confidence, category = classifier.is_scientific_figure(image_path)
            print(f"Image: {image_path}")
            print(f"Is Scientific: {is_scientific}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Category: {category}")
            print()
