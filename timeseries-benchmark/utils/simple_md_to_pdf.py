#!/usr/bin/env python3
"""
Simple markdown to PDF converter using reportlab
"""

import sys
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

def markdown_to_pdf(md_file, pdf_file=None):
    """Convert markdown to PDF"""
    
    if pdf_file is None:
        pdf_file = Path(md_file).with_suffix('.pdf')
    
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Create PDF
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=30,
        alignment=TA_CENTER
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=12,
        spaceBefore=12
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=10,
        spaceBefore=10
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#4a7ba7'),
        spaceAfter=8,
        spaceBefore=8
    ))
    
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Code'],
        fontSize=8,
        leftIndent=20,
        backgroundColor=colors.HexColor('#f5f5f5'),
        borderPadding=10
    ))
    
    # Parse markdown line by line
    lines = md_content.split('\n')
    in_code_block = False
    code_block = []
    in_table = False
    table_data = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End of code block
                code_text = '\n'.join(code_block)
                # Escape for XML
                code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f'<font face="Courier" size="8">{code_text}</font>', styles['Code']))
                story.append(Spacer(1, 0.2*inch))
                code_block = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue
        
        if in_code_block:
            code_block.append(line)
            i += 1
            continue
        
        # Skip image references (can't easily embed)
        if line.strip().startswith('!['):
            story.append(Paragraph(f'<i>[Image: {line[2:line.find("]")]}]</i>', styles['Italic']))
            story.append(Spacer(1, 0.1*inch))
            i += 1
            continue
        
        # Headers
        if line.startswith('# '):
            text = line[2:].strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(text, styles['CustomTitle']))
            story.append(Spacer(1, 0.2*inch))
        elif line.startswith('## '):
            text = line[3:].strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(PageBreak())
            story.append(Paragraph(text, styles['CustomHeading1']))
            story.append(Spacer(1, 0.15*inch))
        elif line.startswith('### '):
            text = line[4:].strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(text, styles['CustomHeading2']))
            story.append(Spacer(1, 0.1*inch))
        elif line.startswith('#### '):
            text = line[5:].strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(text, styles['CustomHeading3']))
            story.append(Spacer(1, 0.1*inch))
        
        # Horizontal rules
        elif line.strip() == '---':
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph('<hr width="100%"/>', styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Bullet lists
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            text = line.strip()[2:]
            text = process_inline_markdown(text)
            story.append(Paragraph(f'• {text}', styles['Normal']))
        
        # Numbered lists
        elif re.match(r'^\d+\.\s', line.strip()):
            text = re.sub(r'^\d+\.\s', '', line.strip())
            text = process_inline_markdown(text)
            story.append(Paragraph(text, styles['Normal']))
        
        # Tables (basic support)
        elif '|' in line and line.strip().startswith('|'):
            # Simple table detection - skip separator lines
            if not all(c in '|-: ' for c in line.strip()):
                row = [cell.strip() for cell in line.strip().split('|')[1:-1]]
                if row and any(row):  # Non-empty row
                    story.append(Paragraph(' | '.join(row), styles['Normal']))
        
        # Regular paragraphs
        elif line.strip():
            text = process_inline_markdown(line.strip())
            story.append(Paragraph(text, styles['BodyText']))
            story.append(Spacer(1, 0.1*inch))
        
        # Empty lines
        else:
            story.append(Spacer(1, 0.1*inch))
        
        i += 1
    
    # Build PDF
    print(f"Generating PDF: {pdf_file}")
    doc.build(story)
    print(f"✅ PDF created successfully: {pdf_file}")
    return pdf_file

def process_inline_markdown(text):
    """Process inline markdown (bold, italic, code)"""
    # Escape XML
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    
    # Italic *text*
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    
    # Code `text`
    text = re.sub(r'`(.+?)`', r'<font face="Courier" size="9">\1</font>', text)
    
    # Links [text](url) - just show text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'<u>\1</u>', text)
    
    return text

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python simple_md_to_pdf.py <markdown_file> [output_pdf]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    markdown_to_pdf(md_file, pdf_file)
