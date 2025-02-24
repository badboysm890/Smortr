import pdf2image
from PIL import ImageChops
import io
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from PIL import Image, ImageChops, ImageDraw
import numpy as np
from imagehash import average_hash
import os
import uuid
from typing import List, Dict
from pydantic import BaseModel

app = FastAPI()

# Create a temporary directory for storing comparison results
TEMP_DIR = "/tmp/pdf_comparisons"
os.makedirs(TEMP_DIR, exist_ok=True)

class ImageCompareRequest(BaseModel):
    image1: str  # base64 encoded image
    image2: str  # base64 encoded image

def are_images_similar(img1, img2, threshold=5):
    hash1 = average_hash(img1)
    hash2 = average_hash(img2)
    difference = hash1 - hash2
    return difference <= threshold

def find_differences(img1, img2):
    diff = ImageChops.difference(img1, img2)
    thresh = 30
    diff_arr = np.array(diff)
    diff_regions = np.where(diff_arr > thresh)
    
    if len(diff_regions[0]) == 0:
        return None, 0
    
    # Calculate difference percentage
    total_pixels = diff_arr.size / 3  # Divide by 3 for RGB channels
    diff_pixels = len(diff_regions[0])
    diff_percentage = (diff_pixels / total_pixels) * 100
    
    # Find bounding boxes of differences
    boxes = []
    x_coords = diff_regions[1]
    y_coords = diff_regions[0]
    if len(x_coords) > 0:
        left, top = min(x_coords), min(y_coords)
        right, bottom = max(x_coords), max(y_coords)
        boxes.append((left, top, right, bottom))
    
    return boxes, diff_percentage

def convert_pdf_to_images(pdf_data):
    pdf_stream = io.BytesIO(pdf_data)
    return pdf2image.convert_from_bytes(pdf_stream.getvalue())

@app.post("/compare-pdfs")
async def compare_pdfs(
    pdf1: UploadFile = File(..., description="First PDF file"),
    pdf2: UploadFile = File(..., description="Second PDF file")
):
    if not pdf1.content_type == "application/pdf" or not pdf2.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Both files must be PDFs")
    
    try:
        # Create unique directory for this comparison
        comparison_id = str(uuid.uuid4())
        comparison_dir = os.path.join(TEMP_DIR, comparison_id)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Read PDF files directly from uploads
        pdf1_content = await pdf1.read()
        pdf2_content = await pdf2.read()
        
        # Convert PDFs to images
        images1 = pdf2image.convert_from_bytes(pdf1_content)
        images2 = pdf2image.convert_from_bytes(pdf2_content)
        
        if len(images1) != len(images2):
            return {"error": "PDFs have different number of pages"}
        
        results = []
        for i in range(len(images1)):
            if not are_images_similar(images1[i], images2[i]):
                results.append({
                    "page": i + 1,
                    "error": "Pages are too different to compare"
                })
                continue
            
            boxes, diff_percentage = find_differences(images1[i], images2[i])
            
            if boxes is None:
                results.append({
                    "page": i + 1,
                    "message": "No differences found",
                    "difference_percentage": 0
                })
                continue
            
            # Draw rectangles around differences
            img1_marked = images1[i].copy()
            img2_marked = images2[i].copy()
            draw1 = ImageDraw.Draw(img1_marked)
            draw2 = ImageDraw.Draw(img2_marked)
            
            for box in boxes:
                draw1.rectangle(box, outline="red", width=2)
                draw2.rectangle(box, outline="red", width=2)
            
            # Save marked images as files
            img1_path = os.path.join(comparison_dir, f"page_{i+1}_doc1.png")
            img2_path = os.path.join(comparison_dir, f"page_{i+1}_doc2.png")
            img1_marked.save(img1_path)
            img2_marked.save(img2_path)
            
            results.append({
                "page": i + 1,
                "difference_percentage": round(diff_percentage, 2),
                "image1_url": f"/images/{comparison_id}/page_{i+1}_doc1.png",
                "image2_url": f"/images/{comparison_id}/page_{i+1}_doc2.png"
            })
        
        return {"comparison_id": comparison_id, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/images/{comparison_id}/{image_name}")
async def get_comparison_image(comparison_id: str, image_name: str):
    image_path = os.path.join(TEMP_DIR, comparison_id, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
