#!/usr/bin/env python3
"""Simple markdown to PDF converter using markdown2 and weasyprint."""

import sys
from pathlib import Path
import markdown2
from weasyprint import HTML, CSS

def convert_md_to_pdf(input_file, output_file):
    """Convert markdown file to PDF."""

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    try:
        print(f"Reading {input_file}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        print("Converting markdown to HTML...")
        html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks', 'toc'])

        # Wrap in HTML document with styling
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    color: #1a1a1a;
                    border-bottom: 3px solid #0066cc;
                    padding-bottom: 10px;
                    page-break-after: avoid;
                }}
                h2 {{
                    color: #0066cc;
                    margin-top: 30px;
                    page-break-after: avoid;
                }}
                h3 {{
                    color: #004499;
                    page-break-after: avoid;
                }}
                code {{
                    background: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                pre {{
                    background: #f4f4f4;
                    padding: 12px;
                    border-radius: 5px;
                    overflow-x: auto;
                    border-left: 4px solid #0066cc;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 15px 0;
                }}
                table th, table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                table th {{
                    background-color: #f4f4f4;
                    font-weight: bold;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 15px 0;
                }}
                blockquote {{
                    border-left: 4px solid #0066cc;
                    margin: 0;
                    padding-left: 15px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        print("Converting HTML to PDF...")
        HTML(string=html_doc).write_pdf(output_path)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Success! Created {output_file} ({size_mb:.2f} MB)")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "IMPLEMENTATION_REPORT.md"
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.md', '.pdf')

    success = convert_md_to_pdf(input_file, output_file)
    sys.exit(0 if success else 1)
