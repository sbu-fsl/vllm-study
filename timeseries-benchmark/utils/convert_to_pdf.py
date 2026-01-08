#!/usr/bin/env python3
"""
Convert PHASE2_EXTENDED_REPORT.md to PDF with plots embedded
Uses markdown2 + pdfkit (or reportlab as fallback)
"""
import os
import sys

# Try multiple methods for PDF generation
def convert_with_weasyprint():
    """Method 1: WeasyPrint (best quality, handles images well)"""
    try:
        import markdown
        from weasyprint import HTML, CSS
        import re
        
        print("Using WeasyPrint for PDF generation...")
        
        # Read markdown
        with open('PHASE2_EXTENDED_REPORT.md', 'r') as f:
            md_content = f.read()
        
        # Fix image paths - convert relative paths to absolute paths
        # Pattern: ![alt text](plots/image.png) -> ![alt text](file:///full/path/plots/image.png)
        current_dir = os.path.abspath(os.path.dirname(__file__))
        def fix_image_path(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            if not img_path.startswith('http') and not img_path.startswith('file://'):
                abs_path = os.path.join(current_dir, img_path)
                if os.path.exists(abs_path):
                    return f'![{alt_text}](file://{abs_path})'
            return match.group(0)
        
        md_content = re.sub(r'!\[(.*?)\]\((.*?)\)', fix_image_path, md_content)
        
        # Convert markdown to HTML with extensions for tables and code
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'codehilite', 'nl2br']
        )
        
        # Add CSS styling
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: letter;
                    margin: 2.5cm;
                }}
                body {{
                    font-family: 'DejaVu Sans', Arial, sans-serif;
                    font-size: 11pt;
                    line-height: 1.6;
                    color: #333;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    page-break-before: always;
                }}
                h1:first-of-type {{
                    page-break-before: avoid;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #555;
                    margin-top: 20px;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                pre {{
                    background-color: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    font-size: 9pt;
                    page-break-inside: avoid;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 6px;
                    text-align: left;
                    word-wrap: break-word;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                img {{
                    max-width: 90%;
                    max-height: 500px;
                    height: auto;
                    display: block;
                    margin: 15px auto;
                    page-break-inside: avoid;
                }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    padding-left: 20px;
                    margin: 20px 0;
                    color: #555;
                    font-style: italic;
                }}
                .page-break {{
                    page-break-after: always;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        HTML(string=html_with_style).write_pdf('PHASE2_EXTENDED_REPORT.pdf')
        print("✅ PDF generated successfully: PHASE2_EXTENDED_REPORT.pdf")
        return True
        
    except ImportError:
        print("❌ WeasyPrint not available")
        return False
    except Exception as e:
        print(f"❌ WeasyPrint failed: {e}")
        return False

def convert_with_markdown2pdf():
    """Method 2: markdown2pdf (simpler, but may not handle images as well)"""
    try:
        import subprocess
        print("Trying markdown-pdf command...")
        result = subprocess.run(
            ['markdown-pdf', 'PHASE2_EXTENDED_REPORT.md', '-o', 'PHASE2_EXTENDED_REPORT.pdf'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ PDF generated successfully: PHASE2_EXTENDED_REPORT.pdf")
            return True
        else:
            print(f"❌ markdown-pdf failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ markdown-pdf not available")
        return False
    except Exception as e:
        print(f"❌ markdown-pdf failed: {e}")
        return False

def convert_with_pandoc():
    """Method 3: Pandoc (most reliable if installed)"""
    try:
        import subprocess
        print("Trying pandoc...")
        result = subprocess.run([
            'pandoc',
            'PHASE2_EXTENDED_REPORT.md',
            '-o', 'PHASE2_EXTENDED_REPORT.pdf',
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '--highlight-style=tango'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PDF generated successfully: PHASE2_EXTENDED_REPORT.pdf")
            return True
        else:
            print(f"❌ Pandoc failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Pandoc not available")
        return False
    except Exception as e:
        print(f"❌ Pandoc failed: {e}")
        return False

def main():
    print("="*70)
    print("Converting PHASE2_EXTENDED_REPORT.md to PDF")
    print("="*70)
    print()
    
    # Try methods in order of preference
    methods = [
        ("WeasyPrint", convert_with_weasyprint),
        ("Pandoc", convert_with_pandoc),
        ("markdown-pdf", convert_with_markdown2pdf),
    ]
    
    for name, method in methods:
        print(f"\n--- Attempting conversion with {name} ---")
        if method():
            print(f"\n{'='*70}")
            print(f"✅ SUCCESS! PDF generated with {name}")
            print(f"{'='*70}")
            print(f"\nOutput: PHASE2_EXTENDED_REPORT.pdf")
            print(f"Size: {os.path.getsize('PHASE2_EXTENDED_REPORT.pdf') / 1024:.1f} KB")
            return 0
    
    print(f"\n{'='*70}")
    print("❌ All conversion methods failed!")
    print("="*70)
    print("\nPlease install one of the following:")
    print("  1. WeasyPrint: pip install weasyprint markdown")
    print("  2. Pandoc: apt-get install pandoc texlive-xelatex")
    print("  3. markdown-pdf: npm install -g markdown-pdf")
    print("\nAlternatively, use an online converter:")
    print("  - https://www.markdowntopdf.com/")
    print("  - https://markdown-pdf.herokuapp.com/")
    return 1

if __name__ == "__main__":
    sys.exit(main())
