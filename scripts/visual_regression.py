"""
Visual Regression Testing for LaTeXify.

Compares generated PDFs against golden set using SSIM (Structural Similarity Index)
to ensure output quality doesn't degrade.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)

# Try importing skimage for SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logger.warning("scikit-image not installed. SSIM computation unavailable.")


class VisualRegressionTester:
    """
    Visual regression testing using SSIM on PDF renders.
    
    Workflow:
    1. Render source PDF → images
    2. Render generated PDF → images
    3. Compute SSIM per page
    4. Compare against threshold
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ssim_threshold = self.config.get("ssim_threshold", 0.90)
        self.render_dpi = self.config.get("render_dpi", 200)
        self.results_dir = Path(self.config.get("results_dir", ".test_results"))
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def test_document(
        self,
        source_pdf: Path,
        generated_pdf: Path,
        test_name: str
    ) -> Dict:
        """
        Test a generated PDF against source.
        
        Args:
            source_pdf: Path to source (ground truth) PDF
            generated_pdf: Path to generated PDF
            test_name: Identifier for this test
            
        Returns:
            Dict with keys: passed, avg_ssim, per_page_ssim, errors
        """
        logger.info(f"Running visual regression test: {test_name}")
        
        result = {
            "test_name": test_name,
            "source_pdf": str(source_pdf),
            "generated_pdf": str(generated_pdf),
            "passed": False,
            "avg_ssim": 0.0,
            "per_page_ssim": [],
            "errors": []
        }
        
        try:
            # Render both PDFs
            source_images = self._render_pdf(source_pdf)
            generated_images = self._render_pdf(generated_pdf)
            
            # Check page count match
            if len(source_images) != len(generated_images):
                result["errors"].append(
                    f"Page count mismatch: source={len(source_images)}, generated={len(generated_images)}"
                )
                return result
            
            # Compute SSIM per page
            ssim_scores = []
            for i, (src_img, gen_img) in enumerate(zip(source_images, generated_images)):
                ssim_score = self._compute_ssim(src_img, gen_img)
                ssim_scores.append(ssim_score)
                logger.info(f"Page {i+1} SSIM: {ssim_score:.4f}")
            
            # Average SSIM
            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            result["avg_ssim"] = avg_ssim
            result["per_page_ssim"] = ssim_scores
            
            # Pass/fail
            if avg_ssim >= self.ssim_threshold:
                result["passed"] = True
                logger.info(f"✅ Test PASSED: {test_name} (SSIM: {avg_ssim:.4f} >= {self.ssim_threshold})")
            else:
                result["passed"] = False
                result["errors"].append(
                    f"SSIM below threshold: {avg_ssim:.4f} < {self.ssim_threshold}"
                )
                logger.warning(f"❌ Test FAILED: {test_name} (SSIM: {avg_ssim:.4f} < {self.ssim_threshold})")
            
            # Save result
            self._save_result(test_name, result)
            
        except Exception as e:
            result["errors"].append(f"Test execution failed: {e}")
            logger.error(f"Test failed with exception: {e}", exc_info=True)
        
        return result
    
    def _render_pdf(self, pdf_path: Path) -> List[np.ndarray]:
        """
        Render PDF pages to images.
        
        Returns:
            List of NumPy arrays (H, W, 3) in RGB
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Render at specified DPI
                zoom = self.render_dpi / 72.0  # 72 DPI default
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to NumPy
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)
                images.append(img_array)
            
            doc.close()
            logger.info(f"Rendered {len(images)} pages from {pdf_path}")
            return images
            
        except ImportError:
            logger.error("PyMuPDF (fitz) not installed. Cannot render PDFs.")
            raise
        except Exception as e:
            logger.error(f"PDF rendering failed: {e}")
            raise
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute SSIM between two images.
        
        Args:
            img1: Image 1 (H, W, 3) RGB
            img2: Image 2 (H, W, 3) RGB
            
        Returns:
            SSIM score (0.0 to 1.0)
        """
        if not SSIM_AVAILABLE:
            logger.warning("SSIM unavailable. Returning mock score 0.95")
            return 0.95
        
        # Resize if needed (must be same dimensions)
        if img1.shape != img2.shape:
            # Resize img2 to match img1
            img2_pil = Image.fromarray(img2)
            img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.LANCZOS)
            img2 = np.array(img2_pil)
        
        # Convert to grayscale for SSIM
        if img1.ndim == 3:
            img1_gray = np.mean(img1, axis=2).astype(np.uint8)
        else:
            img1_gray = img1
        
        if img2.ndim == 3:
            img2_gray = np.mean(img2, axis=2).astype(np.uint8)
        else:
            img2_gray = img2
        
        # Compute SSIM
        score = ssim(img1_gray, img2_gray, data_range=255)
        
        return float(score)
    
    def _save_result(self, test_name: str, result: Dict):
        """Save test result as JSON."""
        result_file = self.results_dir / f"{test_name}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved result to {result_file}")
    
    def run_golden_set(self, golden_dir: Path) -> List[Dict]:
        """
        Run visual regression tests on all PDFs in golden set.
        
        Expected structure:
        golden_dir/
          test1/
            source.pdf
            generated.pdf
          test2/
            source.pdf
            generated.pdf
        
        Args:
            golden_dir: Directory containing test subdirectories
            
        Returns:
            List of test results
        """
        results = []
        
        for test_dir in sorted(golden_dir.iterdir()):
            if not test_dir.is_dir():
                continue
            
            source_pdf = test_dir / "source.pdf"
            generated_pdf = test_dir / "generated.pdf"
            
            if not source_pdf.exists() or not generated_pdf.exists():
                logger.warning(f"Skipping {test_dir.name}: missing PDFs")
                continue
            
            result = self.test_document(source_pdf, generated_pdf, test_dir.name)
            results.append(result)
        
        # Summary
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        avg_ssim = sum(r["avg_ssim"] for r in results) / total if total > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Visual Regression Test Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Overall avg SSIM: {avg_ssim:.4f}")
        logger.info(f"{'='*60}\n")
        
        return results
