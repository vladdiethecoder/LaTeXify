"""
Gradio-based debug UI for LaTeXify pipeline.

Provides interactive debugging interface for:
- Uploading PDFs
- Visualizing detected bounding boxes
- Inspecting intermediate outputs
- Downloading JSON manifests
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

logger = logging.getLogger(__name__)

# Try importing Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    gr = None
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not installed. Debug UI unavailable.")


class DebugUI:
    """
    Gradio-based debug interface for LaTeXify.
    """
    
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.current_state = None
        
    def visualize_layout(
        self,
        image_path: Path,
        regions: List[Any]
    ) -> Image.Image:
        """
        Draw bounding boxes on image.
        
        Args:
            image_path: Path to image
            regions: List of LayoutRegion objects
            
        Returns:
            PIL Image with bboxes overlaid
        """
        # Load image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Color map for categories
        color_map = {
            "Text": "#4CAF50",
            "Title": "#2196F3",
            "Table": "#FF9800",
            "Figure": "#9C27B0",
            "Equation_Display": "#F44336",
            "Equation_Inline": "#FF5722",
            "Caption": "#00BCD4",
            "Header": "#607D8B",
            "Footer": "#9E9E9E",
            "List": "#8BC34A",
            "Sidebar": "#FFEB3B"
        }
        
        # Draw each region
        for region in regions:
            bbox = region.bbox
            category = region.category
            confidence = region.confidence
            
            # Get color
            color = color_map.get(category, "#FFFFFF")
            
            # Draw rectangle
            draw.rectangle(
                [(bbox.x1, bbox.y1), (bbox.x2, bbox.y2)],
                outline=color,
                width=3
            )
            
            # Draw label
            label = f"{category} ({confidence:.2f})"
            draw.text(
                (bbox.x1, bbox.y1 - 20),
                label,
                fill=color,
                font=font
            )
        
        return img
    
    def process_pdf_debug(
        self,
        pdf_file: Any,
        show_layout: bool = True,
        show_extractions: bool = True
    ) -> tuple:
        """
        Process PDF and return debug visualization.
        
        Args:
            pdf_file: Gradio UploadedFile
            show_layout: Show bounding boxes
            show_extractions: Show extracted content
            
        Returns:
            (annotated_images, json_manifest, latex_output)
        """
        if not self.pipeline:
            return None, "Pipeline not initialized", ""
        
        # Save uploaded file
        pdf_path = Path(pdf_file.name)
        
        try:
            # Run pipeline with intermediate outputs
            logger.info(f"Processing {pdf_path} in debug mode...")
            
            # TODO: This would integrate with the actual pipeline
            # For now, return placeholder
            
            annotated_images = []
            manifest = {
                "pdf": str(pdf_path),
                "pages": 1,
                "regions": [],
                "extractions": []
            }
            latex_output = "% LaTeX output would appear here"
            
            return (
                annotated_images,
                json.dumps(manifest, indent=2),
                latex_output
            )
            
        except Exception as e:
            logger.error(f"Debug processing failed: {e}", exc_info=True)
            return None, str(e), ""
    
    def launch(self, share: bool = False, port: int = 7860):
        """
        Launch Gradio UI.
        
        Args:
            share: Create public link
            port: Port to run on
        """
        if not GRADIO_AVAILABLE:
            logger.error("Gradio not installed. Cannot launch UI.")
            return
        
        # Create interface
        with gr.Blocks(title="LaTeXify Debug UI") as demo:
            gr.Markdown("# LaTeXify Gen 3.0 - Debug Interface")
            gr.Markdown("Upload a PDF to visualize layout detection and extraction results.")
            
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"]
                    )
                    
                    show_layout = gr.Checkbox(
                        label="Show Layout Detection",
                        value=True
                    )
                    
                    show_extractions = gr.Checkbox(
                        label="Show Extractions",
                        value=True
                    )
                    
                    process_btn = gr.Button("Process PDF", variant="primary")
                
                with gr.Column():
                    image_output = gr.Gallery(
                        label="Annotated Pages",
                        columns=2,
                        height=600
                    )
            
            with gr.Row():
                with gr.Column():
                    json_output = gr.Code(
                        label="JSON Manifest",
                        language="json",
                        lines=20
                    )
                
                with gr.Column():
                    latex_output = gr.Code(
                        label="LaTeX Output",
                        language="latex",
                        lines=20
                    )
            
            # Connect button
            process_btn.click(
                fn=self.process_pdf_debug,
                inputs=[pdf_input, show_layout, show_extractions],
                outputs=[image_output, json_output, latex_output]
            )
            
            gr.Markdown("""
            ## Features
            - **Layout Detection**: Visualize DocLayout-YOLO bounding boxes
            - **Confidence Scores**: See per-region confidence
            - **JSON Manifest**: Download intermediate state
            - **LaTeX Preview**: View generated code
            """)
        
        # Launch
        logger.info(f"Launching Gradio UI on port {port}...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share
        )


def create_debug_app(pipeline=None):
    """
    Factory function to create debug UI.
    
    Args:
        pipeline: LaTeXifyPipeline instance
        
    Returns:
        DebugUI instance
    """
    return DebugUI(pipeline)
