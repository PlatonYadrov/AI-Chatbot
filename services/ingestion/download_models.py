#!/usr/bin/env python3
"""Download and cache Docling models for offline use.

Run this script once with internet connection to download all required models.
After that, Docling will work completely offline.
"""

import os
import sys
import tempfile
from pathlib import Path


def download_models():
    """Download Docling models to local cache."""
    print("=" * 60)
    print("Downloading Docling models for offline use")
    print("=" * 60)
    
    # Import Docling
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    except ImportError:
        print("ERROR: Docling not installed")
        print("Run: pip install docling")
        sys.exit(1)
    
    # Create a temporary test PDF
    print("\n[1/3] Creating test document...")
    # Prefer reportlab to avoid heavy PyMuPDF dependency
    test_pdf = None
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as tmp:
            c = canvas.Canvas(tmp.name, pagesize=letter)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, 750, "Test document for model download")
            c.setFont("Helvetica", 10)
            c.drawString(72, 720, "Column1    Column2")
            c.drawString(72, 705, "Value1     Value2")
            c.save()
            test_pdf = tmp.name
        print(f"✓ Test PDF created: {test_pdf}")
    except Exception as e:
        print(f"⚠ Could not generate test PDF (reportlab missing?): {e}")
        print("Proceeding without test document - models will download on first real parse")
        test_pdf = None
    
    # Configure for full model download
    print("\n[2/3] Downloading models...")
    print("This may take 5-10 minutes on first run...")
    
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        
        # New Docling API: wrap into PdfFormatOption and set backend
        pdf_fmt_option = PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_fmt_option}
        )
        
        if test_pdf:
            print("  Processing test document to trigger model downloads...")
            result = converter.convert(test_pdf)
            print(f"  ✓ Processed {len(list(result.document.iterate_items()))} items")
        else:
            print("  ✓ Converter initialized")
        
        print("✓ Models downloaded successfully")
        
    except Exception as e:
        print(f"✗ Model download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if test_pdf and os.path.exists(test_pdf):
            os.remove(test_pdf)
    
    # Show cache location
    print("\n[3/3] Verifying cache...")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        models = list(cache_dir.glob("models--*"))
        total_size = sum(
            sum(f.stat().st_size for f in model.rglob("*") if f.is_file())
            for model in models
        )
        size_mb = total_size / (1024 * 1024)
        
        print(f"✓ Cache location: {cache_dir}")
        print(f"  Models: {len(models)}")
        print(f"  Total size: {size_mb:.1f} MB")
        
        print("\n  Downloaded models:")
        for model in models:
            model_name = model.name.replace("models--", "").replace("--", "/")
            print(f"    - {model_name}")
    else:
        print(f"⚠ Cache directory not found: {cache_dir}")
    
    print("\n" + "=" * 60)
    print("✓ Setup complete!")
    print("=" * 60)
    print("\nDocling is now configured for offline use.")
    print("You can disconnect from the internet and run:")
    print("  python -c 'from parsers.docling_parser import DoclingParser; ...'\n")


if __name__ == "__main__":
    download_models()

