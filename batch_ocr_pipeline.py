import os
import io
import time
import json
import math
import requests
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]
from PyPDF2 import PdfReader, PdfWriter  # pyright: ignore[reportMissingImports]
from pdfminer.high_level import extract_text  # pyright: ignore[reportMissingImports]

API_KEY = os.getenv("MISTRAL_API_KEY")
API_URL = "https://api.mistral.ai/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# -----------------------------
# Detect text or image pages
# -----------------------------
def is_text_page(pdf_path):
    """Return True if page likely contains text (not scanned image)."""
    try:
        text = extract_text(pdf_path, maxpages=1)
        return bool(text.strip())
    except Exception:
        return False


# -----------------------------
# Split PDF into chunks (5 pages each)
# -----------------------------
def split_pdf(input_pdf, chunk_size=5, output_dir="split_chunks"):
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)
    chunks = math.ceil(total_pages / chunk_size)
    file_paths = []

    for i in range(chunks):
        writer = PdfWriter()
        start = i * chunk_size
        end = min(start + chunk_size, total_pages)
        for j in range(start, end):
            writer.add_page(reader.pages[j])
        path = os.path.join(output_dir, f"chunk_{i+1}.pdf")
        with open(path, "wb") as f:
            writer.write(f)
        file_paths.append(path)

    print(f"Split into {len(file_paths)} chunks of up to {chunk_size} pages.")
    return file_paths


# -----------------------------
# Upload file to Mistral
# -----------------------------
def upload_file(filepath):
    with open(filepath, "rb") as f:
        response = requests.post(
            f"{API_URL}/files",
            headers=HEADERS,
            files={"file": f},
            data={"purpose": "batch"},
        )
    response.raise_for_status()
    return response.json()["id"]


# -----------------------------
# Create NDJSON batch file (only for scanned chunks)
# -----------------------------
def create_ndjson(scanned_file_ids, output_path="ocr_batch.ndjson"):
    with open(output_path, "w") as f:
        for i, fid in enumerate(scanned_file_ids):
            line = {
                "custom_id": f"chunk_{i+1}",
                "body": {"model": "mistral-ocr", "input": fid},
            }
            f.write(json.dumps(line) + "\n")
    print(f"NDJSON file created with {len(scanned_file_ids)} OCR tasks.")
    return output_path


# -----------------------------
# Upload NDJSON file
# -----------------------------
def upload_batch_file(ndjson_path):
    with open(ndjson_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/files",
            headers=HEADERS,
            files={"file": f},
            data={"purpose": "batch"},
        )
    response.raise_for_status()
    return response.json()["id"]


# -----------------------------
# Create Batch Job
# -----------------------------
def create_batch_job(batch_file_id):
    payload = {
        "input_files": [batch_file_id],
        "endpoint": "/v1/ocr",
        "model": "mistral-ocr",
    }
    response = requests.post(
        f"{API_URL}/batch/jobs",
        headers={**HEADERS, "Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    response.raise_for_status()
    job = response.json()
    print(f"Created OCR batch job: {job['id']}")
    return job["id"]


# -----------------------------
# Poll job until completion
# -----------------------------
def wait_for_job(job_id, interval=15):
    print("Waiting for OCR batch job to complete...")
    while True:
        resp = requests.get(f"{API_URL}/batch/jobs/{job_id}", headers=HEADERS)
        data = resp.json()
        status = data["status"]
        print(f"   → Status: {status}")
        if status in ["completed", "failed", "cancelled"]:
            break
        time.sleep(interval)
    return data


# -----------------------------
# Download and merge results
# -----------------------------
def download_results(output_file_id, save_path="ocr_results.ndjson"):
    r = requests.get(f"{API_URL}/files/{output_file_id}/content", headers=HEADERS)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"OCR results downloaded to: {save_path}")

    texts = []
    with open(save_path, "r") as f:
        for line in f:
            result = json.loads(line)
            output_text = (
                result.get("response", {}).get("output_text")
                or result.get("response", {}).get("text")
                or ""
            )
            texts.append(output_text.strip())

    return texts


# -----------------------------
# Main Workflow
# -----------------------------
def main():
    input_pdf = "Chatbot Proposal V1.1.pdf"
    print(f"Starting smart OCR for {input_pdf}")

    # Split into chunks
    chunk_files = split_pdf(input_pdf, chunk_size=5)

    text_pages, scanned_pages = [], []

    print("\nDetecting text-only vs scanned chunks...")
    for path in tqdm(chunk_files):
        if is_text_page(path):
            text_pages.append(path)
        else:
            scanned_pages.append(path)

    print(f"{len(text_pages)} chunks are text-only (no OCR needed).")
    print(f"{len(scanned_pages)} chunks will be OCR processed.\n")

    # Extract text directly from text chunks
    merged_text = []
    for path in text_pages:
        text = extract_text(path)
        merged_text.append(text.strip())

    # Process scanned chunks via Mistral OCR Batch
    if scanned_pages:
        print("Uploading scanned chunks to Mistral...")
        scanned_ids = [upload_file(p) for p in tqdm(scanned_pages)]
        ndjson_path = create_ndjson(scanned_ids)
        batch_file_id = upload_batch_file(ndjson_path)
        job_id = create_batch_job(batch_file_id)
        job_info = wait_for_job(job_id)

        if job_info.get("output_files"):
            output_file_id = job_info["output_files"][0]
            ocr_texts = download_results(output_file_id)
            merged_text.extend(ocr_texts)
        else:
            print("No OCR output files found!")

    # Merge all text
    final_output = "\n\n".join(merged_text)
    with open("final_smart_ocr_output.txt", "w") as f:
        f.write(final_output)

    print("\nSmart OCR completed successfully!")
    print("Final merged text saved to: final_smart_ocr_output.txt")


if __name__ == "__main__":
    main()
