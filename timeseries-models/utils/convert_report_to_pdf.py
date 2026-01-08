#!/usr/bin/env python3
"""
Convert markdown report to PDF using markdown2 and weasyprint or pdfkit
"""

import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path('/home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2/vllm_report')
MD_FILE = REPORT_DIR / 'vllm_anomaly_report.md'
PDF_FILE = REPORT_DIR / 'vllm_anomaly_report.pdf'

def try_pandoc():
    """Try converting with pandoc"""
    try:
        print("Attempting conversion with pandoc...")
        subprocess.run([
            'pandoc', str(MD_FILE),
            '-o', str(PDF_FILE),
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt'
        ], check=True)
        print(f"✅ Successfully created PDF with pandoc: {PDF_FILE}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Pandoc failed: {e}")
        return False

def try_grip():
    """Try converting with grip (GitHub markdown)"""
    try:
        print("Attempting conversion with grip...")
        # Convert markdown to HTML
        html_file = REPORT_DIR / 'vllm_anomaly_report.html'
        subprocess.run([
            'grip', str(MD_FILE),
            '--export', str(html_file)
        ], check=True, timeout=30)
        
        # Convert HTML to PDF with wkhtmltopdf
        subprocess.run([
            'wkhtmltopdf',
            '--enable-local-file-access',
            str(html_file),
            str(PDF_FILE)
        ], check=True)
        
        print(f"✅ Successfully created PDF with grip: {PDF_FILE}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"❌ Grip failed: {e}")
        return False

def try_markdown_pdf():
    """Try with markdown-pdf npm package"""
    try:
        print("Attempting conversion with markdown-pdf...")
        subprocess.run([
            'markdown-pdf',
            str(MD_FILE),
            '-o', str(PDF_FILE)
        ], check=True)
        print(f"✅ Successfully created PDF with markdown-pdf: {PDF_FILE}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ markdown-pdf failed: {e}")
        return False

def try_python_markdown():
    """Try with Python markdown + weasyprint"""
    try:
        print("Attempting conversion with Python markdown + weasyprint...")
        import markdown
        from weasyprint import HTML, CSS
        
        # Read markdown
        with open(MD_FILE, 'r') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add CSS styling
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 0 20px;
                }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }}
                h3 {{ color: #555; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                ul, ol {{ margin: 15px 0; padding-left: 30px; }}
                li {{ margin: 5px 0; }}
                strong {{ color: #2c3e50; }}
                hr {{ border: none; border-top: 2px solid #95a5a6; margin: 30px 0; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        HTML(string=html_with_style, base_url=str(REPORT_DIR)).write_pdf(str(PDF_FILE))
        
        print(f"✅ Successfully created PDF with Python: {PDF_FILE}")
        return True
    except Exception as e:
        print(f"❌ Python conversion failed: {e}")
        return False

def main():
    print("="*80)
    print("CONVERTING MARKDOWN REPORT TO PDF")
    print("="*80)
    
    if not MD_FILE.exists():
        print(f"❌ Error: Markdown file not found: {MD_FILE}")
        print("Please run vllm_anomaly_detection_report.py first to generate the report.")
        sys.exit(1)
    
    print(f"\nInput: {MD_FILE}")
    print(f"Output: {PDF_FILE}\n")
    
    # Try different conversion methods in order of preference
    methods = [
        ('pandoc', try_pandoc),
        ('Python markdown + weasyprint', try_python_markdown),
        ('grip + wkhtmltopdf', try_grip),
        ('markdown-pdf', try_markdown_pdf)
    ]
    
    for method_name, method_func in methods:
        if method_func():
            print(f"\n{'='*80}")
            print(f"✅ SUCCESS! PDF created using {method_name}")
            print(f"{'='*80}")
            print(f"\nPDF Location: {PDF_FILE}")
            print(f"File size: {PDF_FILE.stat().st_size / 1024:.1f} KB")
            return
    
    print("\n" + "="*80)
    print("❌ ALL CONVERSION METHODS FAILED")
    print("="*80)
    print("\nPlease install one of the following:")
    print("  1. pandoc + texlive-xetex:")
    print("     sudo apt-get install pandoc texlive-xetex")
    print("  2. Python packages:")
    print("     pip install markdown weasyprint")
    print("  3. grip + wkhtmltopdf:")
    print("     pip install grip && sudo apt-get install wkhtmltopdf")
    print("  4. markdown-pdf (Node.js):")
    print("     npm install -g markdown-pdf")
    print(f"\nAlternatively, you can view the report in markdown: {MD_FILE}")
    sys.exit(1)

if __name__ == '__main__':
    main()
