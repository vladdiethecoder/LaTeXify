import arxiv
from pathlib import Path
import logging
import tarfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_main_tex(tar_path: Path, output_tex_path: Path):
    """
    Extract the main .tex file from an arXiv source tarball.
    """
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Find .tex files
            tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex")]
            if not tex_files:
                logger.warning(f"No .tex files found in {tar_path.name}")
                return

            # Heuristic: Find file with \documentclass
            main_member = None
            for member in tex_files:
                f = tar.extractfile(member)
                if f:
                    content = f.read(2048).decode("utf-8", errors="ignore")
                    if "\\documentclass" in content:
                        main_member = member
                        break
            
            if not main_member:
                # Fallback: use the largest .tex file
                main_member = max(tex_files, key=lambda m: m.size)
            
            logger.info(f"Extracted {main_member.name} as main source.")
            f = tar.extractfile(main_member)
            if f:
                output_tex_path.write_bytes(f.read())
                
    except Exception as e:
        logger.error(f"Failed to extract source from {tar_path}: {e}")

def download_golden_set(output_dir: Path, max_results: int = 5):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Search for papers with source files available
    query = "cat:math.NA OR cat:cs.LG" 
    
    logger.info(f"Searching arXiv for {query}...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    for result in client.results(search):
        safe_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '-', '_')).strip().replace(" ", "_")
        pdf_path = output_dir / f"{safe_title}.pdf"
        src_tar_path = output_dir / f"{safe_title}.tar.gz"
        tex_path = output_dir / f"{safe_title}.tex"
        
        if pdf_path.exists() and tex_path.exists():
            logger.info(f"Skipping {pdf_path.name}, already exists.")
            continue
            
        try:
            if not pdf_path.exists():
                logger.info(f"Downloading PDF: {result.title}")
                result.download_pdf(dirpath=output_dir, filename=f"{safe_title}.pdf")
            
            if not src_tar_path.exists() and not tex_path.exists():
                logger.info(f"Downloading Source: {result.title}")
                result.download_source(dirpath=output_dir, filename=f"{safe_title}.tar.gz")
            
            if src_tar_path.exists() and not tex_path.exists():
                extract_main_tex(src_tar_path, tex_path)
                # Optional: cleanup tar
                # src_tar_path.unlink()
            
        except Exception as e:
            logger.error(f"Failed to process {result.title}: {e}")

if __name__ == "__main__":
    download_golden_set(Path("data/golden_set"))