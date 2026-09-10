# OCR Document Intelligence System

A practical document text-extraction application built with **Python, OpenCV, PyTorch and Tesseract OCR**, with a Streamlit web interface.

## Features
- Upload PNG/JPG/JPEG/WEBP document images
- OpenCV preprocessing for noisy/uneven documents
- PyTorch tensor-based document feature stage
- OCR text extraction using Tesseract
- Display extracted text in the browser
- Download OCR output as a `.txt` file
- Simple, interview-friendly architecture

## Architecture
`Document Image → OpenCV Preprocessing → PyTorch Feature Stage → OCR → Extracted Text`

## Installation

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR
Tesseract is an external OCR engine and must be installed separately.

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr
```

Windows: install Tesseract and add it to PATH.

## Run
```bash
streamlit run app.py
```

The browser will open the application.

## Project Structure
```text
OCR_Document_Intelligence_System/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── ocr_pipeline.py
└── sample_data/
```

## Resume Description
**OCR Document Intelligence System | Python, OpenCV, PyTorch, Tesseract OCR, Streamlit**
- Developed a document image processing application that extracts text from uploaded images.
- Applied OpenCV-based grayscale, denoising and adaptive-threshold preprocessing to improve OCR input quality.
- Integrated a PyTorch tensor-based feature-processing stage and Tesseract OCR for automated text extraction.
- Built a Streamlit interface with text preview and downloadable OCR results.

## Interview Talking Points
1. Why OpenCV? For image preprocessing before OCR.
2. Why preprocessing? Cleaner, higher-contrast input generally improves OCR.
3. Why PyTorch? To demonstrate a deep-learning processing stage and provide a path to learned document models.
4. Why Tesseract? It is a mature open-source OCR engine suitable for a practical prototype.
5. What would you improve next? Add a trained document-layout/text-detection model, PDF support, multilingual OCR and confidence scoring.
