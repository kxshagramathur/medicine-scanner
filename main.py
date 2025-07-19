from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import io
import os
import requests

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2Templates for serving HTML files
templates = Jinja2Templates(directory="templates")

# Load your model and processor outside the endpoint to avoid re-loading on each request
# This assumes you have internet access or the model is cached locally
model_id = 'microsoft/Florence-2-large'
try:
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("Attempting to load with CPU if CUDA is not available or for debugging purposes.")
    # Fallback to CPU if CUDA is not available or if there's an error loading to GPU
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def run_example(task_prompt, image_input, text_input=None):
    """
    Modified run_example to accept PIL Image object directly.
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Ensure the model is on the correct device (CUDA if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Process inputs with the correct dtype
    # If the model is loaded with torch.float16, ensure pixel_values are also float16
    # Check the model's dtype for reference, often done by inspecting model.dtype
    # The 'processor' will typically handle the initial conversion, but explicit casting
    # on the tensors before sending to model.generate is safer for consistency.

    # Option 1: Let processor handle and then ensure target dtype
    inputs = processor(text=prompt, images=image_input, return_tensors="pt")
    # Cast pixel_values to the model's expected dtype (e.g., torch.float16)
    # The 'model.dtype' attribute will tell you how the model was loaded.
    target_dtype = model.dtype # Get the dtype the model was loaded with
    inputs_cuda = {k: v.to(device, dtype=target_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
      input_ids=inputs_cuda["input_ids"],
      pixel_values=inputs_cuda["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image_input.width, image_input.height)
    )
    return parsed_answer

# OLLAMA Integration (as in your notebook)
OLLAMA_URL = "http://localhost:11434/api/generate"

def query_gemma(prompt, model_name="gemma3:1b"):
    """Queries the Ollama Gemma model."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running and the model is downloaded."
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama: {e}"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Handles image upload, performs OCR, and extracts information using Gemma.
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Perform OCR
        task_prompt = '<OCR>'
        ocr_result = run_example(task_prompt, image)
        ocr_text = ocr_result.get('<OCR>', 'No OCR text found.')

        # Prepare prompt for Gemma
        sys_prompt = '''Extract the following key information from the OCR result below:
- Batch No. (or Lot No. or B No.)
- MFG Date (Manufacturing Date)
- Exp Date (Expiry Date)
- Price (M.R.P.)
'''
        prompt_for_gemma = sys_prompt + '\n' + ocr_text

        # Query Gemma
        extracted_info = query_gemma(prompt_for_gemma)

        return {"filename": file.filename, "ocr_text": ocr_text, "extracted_info": extracted_info}
    except Exception as e:
        return {"error": str(e)}