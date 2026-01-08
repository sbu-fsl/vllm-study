#!/usr/bin/env python3
"""
Convert markdown report to HTML with embedded images
"""

import markdown
from pathlib import Path
import base64

REPORT_DIR = Path('/home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2/vllm_report')
MD_FILE = REPORT_DIR / 'vllm_anomaly_report.md'
HTML_FILE = REPORT_DIR / 'vllm_anomaly_report.html'

def embed_image(image_path):
    """Convert image to base64 data URL"""
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

def convert_markdown_to_html():
    """Convert markdown to HTML with embedded images"""
    
    # Read markdown
    with open(MD_FILE, 'r') as f:
        md_content = f.read()
    
    # Replace image paths with embedded data
    images = [
        'scenario_comparison.png',
        'metric_analysis.png',
        'performance_analysis.png',
        'detection_heatmaps.png',
        'parameter_impact.png'
    ]
    
    for img in images:
        img_path = REPORT_DIR / img
        if img_path.exists():
            data_url = embed_image(img_path)
            md_content = md_content.replace(f']({img})', f']({data_url})')
    
    # Convert to HTML
    html_content = markdown.markdown(
        md_content, 
        extensions=['tables', 'fenced_code', 'codehilite']
    )
    
    # Add CSS styling
    html_with_style = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM Anomaly Detection Analysis Report</title>
    <style>
        @media print {{
            body {{ font-size: 10pt; }}
            h1 {{ page-break-before: always; }}
            img {{ max-width: 100%; page-break-inside: avoid; }}
            table {{ page-break-inside: avoid; }}
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
            color: #333;
            background-color: #fff;
        }}
        
        h1 {{
            color: #1a1a1a;
            border-bottom: 4px solid #0066cc;
            padding-bottom: 15px;
            margin-top: 40px;
            font-size: 2.5em;
            font-weight: 700;
        }}
        
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 35px;
            font-size: 2em;
            font-weight: 600;
        }}
        
        h3 {{
            color: #34495e;
            margin-top: 25px;
            font-size: 1.5em;
            font-weight: 600;
        }}
        
        h4 {{
            color: #555;
            margin-top: 20px;
            font-size: 1.2em;
            font-weight: 600;
        }}
        
        p {{
            margin: 15px 0;
            text-align: justify;
        }}
        
        strong {{
            color: #1a1a1a;
            font-weight: 600;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 25px 0;
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: linear-gradient(to bottom, #3498db 0%, #2980b9 100%);
            color: white;
            font-weight: 600;
            padding: 14px;
            text-align: left;
            border: 1px solid #2980b9;
        }}
        
        td {{
            border: 1px solid #ddd;
            padding: 12px;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e9ecef;
        }}
        
        code {{
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #c7254e;
        }}
        
        pre {{
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        pre code {{
            background: none;
            padding: 0;
            color: inherit;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            margin: 30px 0;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: block;
        }}
        
        ul, ol {{
            margin: 20px 0;
            padding-left: 40px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 40px 0;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            color: #666;
            font-style: italic;
        }}
        
        .metadata {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #28a745;
        }}
        
        .warning {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        
        .success {{
            background-color: #d4edda;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
            margin: 20px 0;
        }}
        
        @page {{
            margin: 2cm;
        }}
    </style>
</head>
<body>
    {html_content}
    
    <footer style="margin-top: 60px; padding-top: 20px; border-top: 2px solid #e0e0e0; text-align: center; color: #666; font-size: 0.9em;">
        <p>Generated on {Path(MD_FILE).stat().st_mtime}</p>
        <p>Report Directory: {REPORT_DIR}</p>
    </footer>
</body>
</html>
"""
    
    # Write HTML
    with open(HTML_FILE, 'w') as f:
        f.write(html_with_style)
    
    print(f"✅ Successfully created HTML report: {HTML_FILE}")
    print(f"   File size: {HTML_FILE.stat().st_size / 1024:.1f} KB")
    print(f"\n📄 You can:")
    print(f"   1. Open in browser: {HTML_FILE}")
    print(f"   2. Print to PDF from browser (Ctrl+P)")
    print(f"   3. Share the HTML file directly")

if __name__ == '__main__':
    convert_markdown_to_html()
