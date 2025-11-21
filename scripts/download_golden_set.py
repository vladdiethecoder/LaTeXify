import arxiv
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_golden_set(output_dir: Path, max_results: int = 5):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Search for papers with source files available
    # Query: complex math papers
    query = "cat:math.NA OR cat:cs.LG" # Numerical Analysis or ML
    
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
        src_path = output_dir / f"{safe_title}.tar.gz" # arXiv source is usually tar.gz
        
        if pdf_path.exists():
            logger.info(f"Skipping {pdf_path.name}, already exists.")
            continue
            
        try:
            logger.info(f"Downloading PDF: {result.title}")
            result.download_pdf(dirpath=output_dir, filename=f"{safe_title}.pdf")
            
            logger.info(f"Downloading Source: {result.title}")
            result.download_source(dirpath=output_dir, filename=f"{safe_title}.tar.gz")
            
            # Note: Extracting source to get a single .tex file is complex as it might contain many.
            # For Golden Set, manual curation or complex extraction logic is needed.
            # Here we just download the artifacts.
            
        except Exception as e:
            logger.error(f"Failed to download {result.title}: {e}")

if __name__ == "__main__":
    download_golden_set(Path("data/golden_set"))
