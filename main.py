# Author- Kushagra Mathur
# FastPI and ML-powered OCR/Information Extraction API

# This API allows users to upload medicine packaging images, performs OCR using a vision-language model,
# and extracts structured information using AWS Bedrock LLMs.

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import io
import os
import requests
import boto3
from botocore.exceptions import ClientError

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2Templates for serving HTML files
templates = Jinja2Templates(directory="templates")

# Load Florence-2 model and processor for OCR (Vision-Language)
# ------------------------------------------------------------
# These are loaded once at startup for efficiency. May take 5 mins to Load uplon Execution
model_id = 'microsoft/Florence-2-large'
try:
    # Try loading model on GPU with float16 for speed and memory efficiency
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
    Runs Florence-2 model for OCR or vision-language tasks.
    Args:
        task_prompt (str): The prompt for the model (e.g., '<OCR>').
        image_input (PIL.Image): The input image.
        text_input (str, optional): Additional text for the prompt.
    Returns:
        dict: Parsed model output (e.g., OCR text).
    """
    # Build the prompt for Florence-2
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Use GPU if available, else CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare inputs for the model
    inputs = processor(text=prompt, images=image_input, return_tensors="pt")
    # Ensure pixel_values are cast to the model's expected dtype
    target_dtype = model.dtype
    inputs_cuda = {k: v.to(device, dtype=target_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

    # Generate output from the model
    generated_ids = model.generate(
      input_ids=inputs_cuda["input_ids"],
      pixel_values=inputs_cuda["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    # Decode and post-process the output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image_input.width, image_input.height)
    )
    return parsed_answer


def query_bedrock(prompt):
    """
    Queries AWS Bedrock LLM for information extraction from OCR text.
    Args:
        prompt (str): The system prompt and OCR result.
    Returns:
        str: Extracted information in JSON or error message.
    """
    client = boto3.client("bedrock-runtime", region_name="ap-south-1")
    model_id = "apac.amazon.nova-lite-v1:0"
    conversation = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]
    try:
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
        )
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text
    except (ClientError, Exception) as e:
        return f"ERROR: Can't invoke '{model_id}'. Reason: {e}"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML page for the web UI.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image, perform OCR, and extract structured info.
    Steps:
    1. Read and decode the uploaded image.
    2. Run Florence-2 model for OCR.
    3. Build a detailed system prompt for Bedrock LLM.
    4. Query Bedrock for structured extraction.
    5. Return filename, OCR text, and extracted info (JSON).
    """
    try:
        # Step 1: Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Step 2: Perform OCR using Florence-2
        task_prompt = '<OCR>'
        ocr_result = run_example(task_prompt, image)
        ocr_text = ocr_result.get('<OCR>', 'No OCR text found.')

        # Step 3: Build system prompt for Bedrock LLM
        sys_prompt = (
            "You are an intelligent information extractor designed to process noisy OCR text from medicine packaging labels. "
            "Your job is to extract key information into structured JSON using pattern recognition and semantic clues rather than relying only on text proximity.\n\n"
            "You must extract the following fields:\n\n"
            "1. \"batch_number\":\n"
            "   - Typically labeled as \"Batch No\", \"Lot No\", \"B. No\", \"B.No\", or \"Batch\".\n"
            "   - It's an alphanumeric code (like \"B4JNO2\", \"ASH114025AM\").\n"
            "   - Avoid picking purely numeric values unless they match known formats.\n"
            "   - If multiple candidates appear near the label, choose the one with the correct pattern.\n\n"
            "2. \"mfg_date\":\n"
            "   - Manufacturing date, typically labeled as \"MFD\", \"MFG\", \"MFG DATE\", \"MFD. DT.\", etc.\n"
            "   - Format it as \"MMM YYYY\" (e.g., \"AUG 2024\").\n"
            "   - If multiple dates appear and only one is labeled \"EXP\", the earlier date is likely the \"mfg_date\".\n"
            "   - Handle noisy OCR like \"JUN202024\" or \"AUG202023\" → Convert these to \"JUN 2024\", \"AUG 2023\".\n\n"
            "3. \"exp_date\":\n"
            "   - Expiry date, typically labeled as \"EXP\", \"EXPIRY\", \"EXP. DT\", \"EXP DATE\", etc.\n"
            "   - Format it as \"MMM YYYY\" (e.g., \"JUL 2027\").\n"
            "   - If only a relative instruction is given (e.g., \"Use before 36 months from MFD\"), calculate the expiry by adding the duration to the MFG date.\n\n"
            "4. \"price\":\n"
            "   - Extract the maximum retail price (MRP), labeled like \"M.R.P.\", \"Price ₹\", \"MRP per X g\", etc.\n"
            "   - Return only the numeric price (e.g., 157.45 or 550.00).\n"
            "   - Ignore symbols like \"₹\" or \"/10 Capsules\", or any tax statements.\n\n"
            "Special Handling Instructions:\n"
            "- Fix noisy years like \"202024\" → \"2024\", \"202026\" → \"2026\".\n"
            "- If two dates are next to each other like \"JUN202024 JUN 202026\", treat the earlier as \"mfg_date\" and later as \"exp_date\".\n"
            "- Do not rely only on proximity to labels; use intelligent matching of patterns.\n"
            "- If a field cannot be found, return its value as null (e.g., \"mfg_date\": null).\n"
            "- Always return a valid and minimal JSON structure with only these four keys:\n"
            "  \"batch_number\", \"mfg_date\", \"exp_date\", and \"price\".\n\n"
            "Example format:\n"
            "{\n  \"batch_number\": \"ASH114025AM\",\n  \"mfg_date\": \"AUG 2024\",\n  \"exp_date\": \"JUL 2027\",\n  \"price\": \"157.45\"\n}\n\n"
            "OCR result:\n" + ocr_text
        )

        # Step 4: Query Bedrock for structured extraction
        extracted_info = query_bedrock(sys_prompt)

        # Step 5: Return results to frontend
        return {"filename": file.filename, "ocr_text": ocr_text, "extracted_info": extracted_info}
    except Exception as e:
        # Return error message if any step fails
        return {"error": str(e)}
