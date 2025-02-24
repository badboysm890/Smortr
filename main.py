import pdf2image
from PIL import ImageChops
import io
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageChops, ImageDraw
import numpy as np
from imagehash import average_hash
import os
import uuid
from typing import List, Dict
from pydantic import BaseModel
import asyncio

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

# Constants for optimization
MAX_PDF_SIZE = 5 * 1024 * 1024  # 5MB
MAX_PAGES = 10
DPI = 150  # Lower DPI for faster processing

async def process_pdf_chunk(pdf_content, start_page, end_page):
    try:
        images = pdf2image.convert_from_bytes(
            pdf_content,
            first_page=start_page,
            last_page=end_page,
            dpi=DPI
        )
        return images
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

@app.post("/compare-pdfs")
async def compare_pdfs(
    pdf1: UploadFile = File(...),
    pdf2: UploadFile = File(...),
):
    # Validate file sizes
    if (await pdf1.read(MAX_PDF_SIZE + 1)) > MAX_PDF_SIZE:
        raise HTTPException(status_code=400, detail="PDF 1 is too large (max 5MB)")
    if (await pdf2.read(MAX_PDF_SIZE + 1)) > MAX_PDF_SIZE:
        raise HTTPException(status_code=400, detail="PDF 2 is too large (max 5MB)")
    
    # Reset file positions
    await pdf1.seek(0)
    await pdf2.seek(0)
    
    try:
        comparison_id = str(uuid.uuid4())
        comparison_dir = os.path.join(TEMP_DIR, comparison_id)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Read files
        pdf1_content = await pdf1.read()
        pdf2_content = await pdf2.read()
        
        # Get number of pages
        images1 = await process_pdf_chunk(pdf1_content, 1, 1)
        images2 = await process_pdf_chunk(pdf2_content, 1, 1)
        
        if len(images1) > MAX_PAGES or len(images2) > MAX_PAGES:
            raise HTTPException(status_code=400, detail=f"PDFs must not exceed {MAX_PAGES} pages")
        
        results = []
        # Process pages in chunks of 2
        for i in range(0, len(images1), 2):
            chunk_end = min(i + 2, len(images1))
            
            # Process chunks concurrently
            tasks = [
                process_pdf_chunk(pdf1_content, i + 1, chunk_end),
                process_pdf_chunk(pdf2_content, i + 1, chunk_end)
            ]
            chunk_images1, chunk_images2 = await asyncio.gather(*tasks)
            
            for j, (img1, img2) in enumerate(zip(chunk_images1, chunk_images2)):
                page_num = i + j + 1
                
                # Resize images for faster processing
                img1 = img1.resize((int(img1.width/2), int(img1.height/2)))
                img2 = img2.resize((int(img2.width/2), int(img2.height/2)))
                
                if not are_images_similar(img1, img2):
                    results.append({
                        "page": page_num,
                        "error": "Pages too different"
                    })
                    continue
                
                boxes, diff_percentage = find_differences(img1, img2)
                
                if boxes:
                    # Save only pages with differences
                    img1_path = os.path.join(comparison_dir, f"page_{page_num}_doc1.png")
                    img2_path = os.path.join(comparison_dir, f"page_{page_num}_doc2.png")
                    
                    # Optimize image saving
                    img1.save(img1_path, "PNG", optimize=True)
                    img2.save(img2_path, "PNG", optimize=True)
                    
                    results.append({
                        "page": page_num,
                        "difference_percentage": round(diff_percentage, 2),
                        "image1_url": f"/images/{comparison_id}/page_{page_num}_doc1.png",
                        "image2_url": f"/images/{comparison_id}/page_{page_num}_doc2.png"
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
