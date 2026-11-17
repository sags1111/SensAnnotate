"""
test_llava.py
-------------
Quick test of LLaVA with actual image crops
"""

import requests
import base64
import json
from PIL import Image
import numpy as np
import io

def encode_image(image_rgb: np.ndarray) -> str:
    """Convert numpy image to base64 string for LLaVA."""
    pil_img = Image.fromarray(image_rgb)
    # Resize to max 512px for speed
    pil_img.thumbnail((512, 512))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def classify_with_llava(image_rgb: np.ndarray) -> tuple:
    """
    Send image crop to LLaVA and get sensitivity classification.
    Returns: (sensitivity, reason)
    """
    image_b64 = encode_image(image_rgb)
    
    prompt = """You are analyzing an image crop from a manufacturing facility.
Is the main object in this image sensitive (reveals personal identity or confidential business information) or non-sensitive (generic equipment, furniture, structural elements)?

Answer in exactly this format:
SENSITIVITY: sensitive or non-sensitive
REASON: one sentence"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava",
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
    )
    
    result = response.json()["response"].strip()
    print(f"LLaVA response: {result}")
    
    # Parse response
    sensitivity = "non_sensitive"
    reason = "LLaVA classification"
    
    for line in result.split('\n'):
        line_lower = line.lower().strip()
        if line_lower.startswith("sensitivity:"):
            value = line_lower.replace("sensitivity:", "").strip()
            sensitivity = "sensitive" if value.startswith("sensitive") else "non_sensitive"
        if line_lower.startswith("reason:"):
            reason = line.replace("REASON:", "").replace("Reason:", "").strip()
    
    return sensitivity, reason


# Test with real image crops
if __name__ == "__main__":
    from PIL import Image as PILImage
    import numpy as np
    
    # Test images — use your existing test images
    test_images = [
        "/Users/krupali/Downloads/pexels-equalstock-31336007.jpg",
        "/Users/krupali/Downloads/pexels-mazhar-ulazhar-50963217-31352672.jpg",
    ]
    
    for img_path in test_images:
        try:
            img = np.array(PILImage.open(img_path).convert("RGB"))
            print(f"\nTesting: {img_path}")
            sensitivity, reason = classify_with_llava(img)
            print(f"Result: {sensitivity} | {reason}")
        except Exception as e:
            print(f"Error: {e}")