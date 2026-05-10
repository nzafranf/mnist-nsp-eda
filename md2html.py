#!/usr/bin/env python3
"""Convert markdown to HTML, then print to PDF from browser."""

import sys
from pathlib import Path
import markdown2

def convert_md_to_html(input_file, output_file):
    """Convert markdown file to HTML."""

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
        html_content = markdown2.markdown(
            md_content,
            extras=['tables', 'fenced-code-blocks', 'toc', 'smarty']
        )

        # Wrap in HTML document with professional styling
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Implementation Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        @page {{
            margin: 1in;
            size: letter;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            background: #fff;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}

        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
            page-break-after: avoid;
            margin-top: 30px;
            font-size: 2em;
        }}

        h2 {{
            color: #0066cc;
            margin-top: 30px;
            page-break-after: avoid;
            font-size: 1.6em;
        }}

        h3 {{
            color: #004499;
            page-break-after: avoid;
            font-size: 1.3em;
        }}

        h4 {{
            color: #006699;
            font-size: 1.1em;
        }}

        p {{
            margin: 10px 0;
        }}

        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', 'Monaco', monospace;
            font-size: 0.95em;
        }}

        pre {{
            background: #f4f4f4;
            padding: 12px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #0066cc;
            line-height: 1.4;
            margin: 15px 0;
        }}

        pre code {{
            background: none;
            padding: 0;
            border-radius: 0;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            page-break-inside: avoid;
        }}

        table th, table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}

        table th {{
            background-color: #f0f0f0;
            font-weight: bold;
            color: #333;
        }}

        table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            page-break-inside: avoid;
        }}

        blockquote {{
            border-left: 4px solid #0066cc;
            margin: 15px 0;
            padding-left: 15px;
            color: #666;
        }}

        a {{
            color: #0066cc;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}

        li {{
            margin: 5px 0;
        }}

        strong {{
            font-weight: 600;
            color: #1a1a1a;
        }}

        em {{
            font-style: italic;
        }}

        hr {{
            border: none;
            border-top: 2px solid #0066cc;
            margin: 30px 0;
        }}

        .toc {{
            background: #f9f9f9;
            border: 1px solid #ddd;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            page-break-inside: avoid;
        }}

        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}

        .toc li {{
            margin: 5px 0;
        }}

        .toc a {{
            color: #0066cc;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            a {{
                color: #0066cc;
            }}

            h1, h2, h3 {{
                page-break-after: avoid;
            }}

            ul, ol, p, table, blockquote {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

        print(f"Writing HTML to {output_file}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_doc)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Success! Created {output_file} ({size_mb:.2f} MB)")
        print(f"\nTo convert to PDF:")
        print(f"  1. Open {output_file} in your browser (Chrome/Edge recommended)")
        print(f"  2. Press Ctrl+P or File > Print")
        print(f"  3. Select 'Save as PDF' as the printer")
        print(f"  4. Click Save")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "IMPLEMENTATION_REPORT.md"
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.md', '.html')

    success = convert_md_to_html(input_file, output_file)
    sys.exit(0 if success else 1)
