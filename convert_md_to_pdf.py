#!/usr/bin/env python3
"""
Convert IMPLEMENTATION_REPORT.md to PDF using pandoc.

Requirements:
    pip install pypandoc
    AND one of:
    - pandoc (system package)
    - python -m pip install pandoc (slower fallback)

Usage:
    python convert_md_to_pdf.py
    python convert_md_to_pdf.py --input INPUT_FILE.md --output OUTPUT_FILE.pdf
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_pandoc_installed():
    """Check if pandoc is available on system."""
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def convert_with_pandoc(input_file, output_file):
    """Convert markdown to PDF using pandoc command line."""
    cmd = [
        "pandoc",
        str(input_file),
        "-o", str(output_file),
        "--from=markdown",
        "--to=pdf",
        "--pdf-engine=xelatex",  # Use xelatex for better font support
        "-V", "documentclass=article",
        "-V", "geometry=margin=1in",
        "-V", "fontsize=11pt",
        "-V", "linestretch=1.5",
        "--toc",  # Add table of contents
        "--toc-depth=2",
        "--number-sections",  # Number sections
    ]

    try:
        print(f"Converting {input_file} to {output_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False

        output_path = Path(output_file)
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Successfully created {output_file} ({size_mb:.2f} MB)")
            return True
        return False

    except FileNotFoundError:
        print("Error: pandoc not found. Install it with: conda install -c conda-forge pandoc")
        return False


def convert_with_pypandoc(input_file, output_file):
    """Convert markdown to PDF using pypandoc."""
    try:
        import pypandoc
    except ImportError:
        print("Error: pypandoc not installed. Install with: pip install pypandoc")
        return False

    try:
        print(f"Converting {input_file} to {output_file}...")

        with open(input_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        pdf_output = pypandoc.convert_text(
            md_content,
            'pdf',
            format='md',
            extra_args=[
                '--pdf-engine=xelatex',
                '-V', 'documentclass=article',
                '-V', 'geometry=margin=1in',
                '-V', 'fontsize=11pt',
                '--toc',
                '--toc-depth=2',
                '--number-sections',
            ]
        )

        with open(output_file, 'wb') as f:
            f.write(pdf_output)

        output_path = Path(output_file)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Successfully created {output_file} ({size_mb:.2f} MB)")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Markdown to PDF",
        epilog="Example: python convert_md_to_pdf.py --input report.md --output report.pdf"
    )
    parser.add_argument(
        "--input", "-i",
        default="IMPLEMENTATION_REPORT.md",
        help="Input markdown file (default: IMPLEMENTATION_REPORT.md)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output PDF file (default: input filename with .pdf extension)"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    if args.output is None:
        output_file = input_file.with_suffix('.pdf')
    else:
        output_file = Path(args.output)

    # Try pandoc first (better), then pypandoc
    if check_pandoc_installed():
        success = convert_with_pandoc(input_file, output_file)
    else:
        print("pandoc not found on system, trying pypandoc...")
        success = convert_with_pypandoc(input_file, output_file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
