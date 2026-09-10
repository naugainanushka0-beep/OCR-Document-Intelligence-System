import streamlit as st
from pathlib import Path
from src.ocr_pipeline import process_image

st.set_page_config(page_title="OCR Document Intelligence", page_icon="📄", layout="wide")
st.title("📄 OCR Document Intelligence System")
st.caption("Computer Vision + PyTorch + OCR document text extraction")

uploaded = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    data = uploaded.getvalue()

    Path("sample_data").mkdir(exist_ok=True)

    temp = Path("sample_data") / uploaded.name
    temp.write_bytes(data)

    col1, col2 = st.columns(2)
    with col1:
        st.image(data, caption="Uploaded document", use_container_width=True)

    with st.spinner("Processing document..."):
        result = process_image(str(temp))

    with col2:
        st.subheader("Extracted Text")
        if result["text"].strip():
            st.text_area("OCR Result", result["text"], height=420)
            st.download_button(
                "⬇️ Download text",
                result["text"],
                file_name=f"{Path(uploaded.name).stem}_extracted.txt",
                mime="text/plain"
            )
        else:
            st.warning("No readable text was detected. Try a clearer image.")

    with st.expander("View preprocessing details"):
        st.write("Image shape:", result["shape"])
        st.write("Preprocessing:", "grayscale → denoise → adaptive threshold")
