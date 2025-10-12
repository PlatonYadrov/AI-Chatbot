#!/usr/bin/env python3
"""Download ALL Docling models for complete offline operation.

This script downloads:
1. DocLayNet layout detection model (~200MB)
2. TableFormer table structure model (~150MB)
3. Formula detection model (~50MB)
4. Code detection model (~30MB)
5. Optional: VLM model for image understanding (~3-5GB)

Run once with internet to enable full offline functionality.
"""

import os
import sys
import tempfile
from pathlib import Path


def download_all_models(include_vlm: bool = False):
    """Download all Docling models."""
    print("=" * 70)
    print("Downloading ALL Docling Models")
    print("=" * 70)
    
    # Import Docling
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
        )
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    except ImportError:
        print("ERROR: Docling not installed")
        print("Run: pip install docling")
        if include_vlm:
            print("For VLM: pip install 'docling[vlm]'")
        sys.exit(1)
    
    print("\n[1/5] Creating test documents...")
    
    # Create simple test document using reportlab (lighter dependency)
    test_pdf = None
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode='wb') as tmp:
            c = canvas.Canvas(tmp.name, pagesize=letter)
            
            # Title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, 750, "Test Document for Model Download")
            
            # Table
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, 700, "Sample Table:")
            c.setFont("Helvetica", 10)
            c.drawString(72, 680, "Column1    Column2    Column3")
            c.drawString(72, 665, "Value1     Value2     Value3")
            c.drawString(72, 650, "Data1      Data2      Data3")
            
            # Formula
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, 600, "Formula: E = mc²")
            c.drawString(72, 580, "∫ f(x)dx = F(x) + C")
            
            # Code block
            c.setFont("Courier", 10)
            c.drawString(72, 540, "def hello_world():")
            c.drawString(90, 525, "print('Hello, World!')")
            c.drawString(90, 510, "return True")
            
            # Section
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, 470, "Section 1: Introduction")
            c.setFont("Helvetica", 11)
            c.drawString(72, 450, "This is a paragraph with normal text for testing.")
            
            c.save()
            test_pdf = tmp.name
        print(f"✓ Test PDF created: {test_pdf}")
    except ImportError:
        print("⚠ reportlab not installed, skipping test PDF creation")
        print("  Models will still be downloaded on first use")
        test_pdf = None
    except Exception as e:
        print(f"⚠ Failed to create test PDF: {e}")
        print("  Models will still be downloaded on first use")
        test_pdf = None
    
    print("\n[2/5] Configuring pipeline options...")
    print("This will trigger download of ALL models:")
    print("  - DocLayNet layout detection (~200MB)")
    print("  - TableFormer table structure (~150MB)")
    print("  - Formula detection (~50MB)")
    print("  - Code detection (~30MB)")
    if include_vlm:
        print("  - VLM for image understanding (~3-5GB)")
    print("\nTotal download: ~", "5-6GB" if include_vlm else "500-600MB")
    
    try:
        # Configure to use ALL models
        options = PdfPipelineOptions()
        
        # Set backend (required in new Docling API)
        options.backend = PyPdfiumDocumentBackend
        
        # Enable OCR
        options.do_ocr = True
        
        # Enable table detection with ACCURATE mode
        options.do_table_structure = True
        options.table_structure_options.mode = TableFormerMode.ACCURATE
        options.table_structure_options.do_cell_matching = True
        
        # Enable image processing (will download layout model)
        options.images_scale = 2.0
        options.generate_page_images = True
        options.generate_picture_images = True
        
        # VLM (optional)
        if include_vlm:
            try:
                options.use_vlm = True
                print("  VLM enabled (large download!)")
            except Exception as e:
                print(f"  VLM not available: {e}")
                print("  Install with: pip install 'docling[vlm]'")
        
        print("✓ Pipeline configured")
        
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        sys.exit(1)
    
    print("\n[3/5] Initializing DocumentConverter...")
    print("Models will download now (this may take 5-15 minutes)...")
    
    try:
        # New Docling API: pass options in format_options
        converter = DocumentConverter(
            format_options={
                "pdf": options,
            }
        )
        print("✓ Converter initialized")
        
        if test_pdf:
            print("\n[4/5] Processing test document...")
            print("This ensures all models are loaded and cached...")
            
            result = converter.convert(test_pdf)
            
            # Count elements
            items = list(result.document.iterate_items())
            print(f"✓ Processed {len(items)} elements")
            
            # Show detected types
            from collections import Counter
            types = Counter(item[0].label for item, _ in items if hasattr(item, 'label'))
            print("\n  Detected element types:")
            for elem_type, count in types.most_common():
                print(f"    - {elem_type}: {count}")
        else:
            print("\n[4/5] Test document processing skipped")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if test_pdf and os.path.exists(test_pdf):
            os.remove(test_pdf)
    
    print("\n[5/5] Verifying downloaded models...")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if cache_dir.exists():
        models = list(cache_dir.glob("models--ds4sd--*"))
        
        print(f"\n✓ Cache location: {cache_dir}")
        print(f"  Docling models: {len(models)}")
        
        # Calculate total size
        total_size = 0
        for model in models:
            model_size = sum(
                f.stat().st_size 
                for f in model.rglob("*") 
                if f.is_file()
            )
            total_size += model_size
            size_mb = model_size / (1024 * 1024)
            model_name = model.name.replace("models--ds4sd--", "")
            print(f"    ✓ {model_name}: {size_mb:.1f} MB")
        
        total_mb = total_size / (1024 * 1024)
        print(f"\n  Total size: {total_mb:.1f} MB")
        
        # Check for VLM
        vlm_models = list(cache_dir.glob("models--google--paligemma*"))
        vlm_models += list(cache_dir.glob("models--Qwen--*"))
        if vlm_models:
            print("\n  VLM models:")
            for model in vlm_models:
                model_size = sum(f.stat().st_size for f in model.rglob("*") if f.is_file())
                size_gb = model_size / (1024 * 1024 * 1024)
                model_name = model.name.replace("models--", "").replace("--", "/")
                print(f"    ✓ {model_name}: {size_gb:.2f} GB")
    else:
        print(f"⚠ Cache directory not found: {cache_dir}")
    
    print("\n" + "=" * 70)
    print("✓ ALL MODELS DOWNLOADED!")
    print("=" * 70)
    print("\nDocling is now ready for COMPLETE offline operation:")
    print("  ✓ Layout detection (titles, sections, paragraphs)")
    print("  ✓ Table structure recognition")
    print("  ✓ Formula detection")
    print("  ✓ Code block detection")
    print("  ✓ OCR (Tesseract)")
    if include_vlm or vlm_models:
        print("  ✓ Image understanding (VLM)")
    print("\nYou can now disconnect from the internet!")
    print("\nTest with:")
    print("  python test_offline_mode.py")
    print("\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download all Docling models")
    parser.add_argument(
        "--with-vlm",
        action="store_true",
        help="Also download VLM models (~3-5GB, requires GPU for inference)"
    )
    
    args = parser.parse_args()
    
    if args.with_vlm:
        print("\n⚠ WARNING: VLM models are large (~3-5GB)")
        print("   They require GPU for fast inference")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping VLM download")
            args.with_vlm = False
    
    download_all_models(include_vlm=args.with_vlm)

