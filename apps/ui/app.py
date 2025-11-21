import streamlit as st
from pathlib import Path
import sys
import os

# Add src to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "src"))

from latexify.core.pipeline import LaTeXifyPipeline
from omegaconf import OmegaConf

st.set_page_config(page_title="LaTeXify Demo", layout="wide")

st.title("LaTeXify: High-Fidelity PDF to LaTeX")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Mock config
cfg = OmegaConf.create({
    "pipeline": {
        "ingestion": {"dpi": 300},
        "layout": {"model": "yolov10n.pt"},
        "ocr": {"lang": "en"},
        "refinement": {"enabled": True}
    }
})

if uploaded_file:
    st.write(f"Processing: {uploaded_file.name}")
    
    # Save temp
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    pdf_path = temp_dir / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getvalue())
    
    if st.button("Run Pipeline"):
        with st.spinner("Processing... (This utilizes RTX 5090 logic, might run slow on CPU)"):
            try:
                pipeline = LaTeXifyPipeline(cfg)
                latex_output = pipeline.process(pdf_path)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Generated LaTeX")
                    st.code(latex_output, language="latex")
                    
                with col2:
                    st.subheader("Rendered Preview (Approximate)")
                    st.markdown(latex_output) # Basic markdown render
                    
                st.download_button("Download .tex", latex_output, file_name="output.tex")
                
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
