---
name: "pdf"
description: "Comprehensive PDF processing: extract text, tables, images, and layout-aware content from PDF files. Invoke when user asks to read/analyze PDF content, extract tables from PDF, compare PDF content, or convert PDF to other formats."
---

# PDF Processing Skill

This skill provides comprehensive PDF extraction and analysis capabilities. It uses multiple tools in a layered approach to maximize extraction fidelity and avoid missing content.

## Available Tools

### 1. Python Libraries (Preferred - highest fidelity)
- **pdfplumber** (0.11.10): Best for tables and structured content
- **PyMuPDF / fitz** (1.27.2.3): Best for text, images, and rendering
- Both available at `/home/luwa/.conda/envs/dsclr/bin/python`

### 2. Poppler CLI Tools (Fallback)
- `pdftotext -layout`: Layout-preserving text extraction
- `pdftotext -table`: Table-aware extraction
- `pdfimages`: Extract embedded images
- `pdftoppm`: Render pages to images (PNG/JPEG)
- `pdftohtml`: Convert to HTML (preserves layout)
- `pdfinfo`: Get PDF metadata and structure
- `pdftocairo`: Advanced rendering (PNG/SVG/PS/PDF)

## Processing Strategy (Use in Order)

### Layer 1: Metadata & Structure
```bash
/home/luwa/.conda/envs/dsclr/bin/python -c "
import fitz
doc = fitz.open('/path/to/file.pdf')
print(f'Pages: {len(doc)}')
for i, page in enumerate(doc):
    print(f'Page {i+1}: {page.rect}')
    # Get text blocks with positions
    for block in page.get_text('dict')['blocks']:
        if 'lines' in block:
            print(f'  Block at y={block[\"bbox\"][1]:.0f}')
"
```

### Layer 2: Table Extraction (HIGH PRIORITY - tables often lost in plain text)
```bash
/home/luwa/.conda/envs/dsclr/bin/python -c "
import pdfplumber
with pdfplumber.open('/path/to/file.pdf') as pdf:
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        for j, table in enumerate(tables):
            print(f'=== Page {i+1} Table {j+1} ===')
            for row in table:
                print(' | '.join([str(c) if c else '' for c in row]))
"
```

### Layer 3: Layout-Aware Text
```bash
# Preserves spatial layout (good for multi-column)
pdftotext -layout "/path/to/file.pdf" /tmp/output.txt

# Table-aware (newer feature, sometimes better for tables)
pdftotext -table "/path/to/file.pdf" /tmp/output.txt
```

### Layer 4: Structured Text with Positions (for complex layouts)
```bash
/home/luwa/.conda/envs/dsclr/bin/python -c "
import fitz
doc = fitz.open('/path/to/file.pdf')
for i, page in enumerate(doc):
    # Get text with structure
    text = page.get_text('text')
    print(f'--- Page {i+1} ---')
    print(text)
    # Also get blocks with coordinates for understanding layout
    blocks = page.get_text('blocks')
    # blocks are sorted by position
"
```

### Layer 5: Render to Images (for visual verification)
```bash
# Render specific page to PNG for visual inspection
pdftoppm -png -f 5 -l 5 -r 200 "/path/to/file.pdf" /tmp/page

# Or use PyMuPDF for higher quality
/home/luwa/.conda/envs/dsclr/bin/python -c "
import fitz
doc = fitz.open('/path/to/file.pdf')
page = doc[4]  # 0-indexed page 5
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
pix.save('/tmp/page5.png')
"
```

## Best Practices

### For Tables (CRITICAL - most error-prone)
1. **ALWAYS use pdfplumber first** for table extraction
2. Cross-verify with `pdftotext -layout` output
3. For complex tables, render the page to image and visually inspect
4. Check table extraction against surrounding text context

### For Multi-Column Layouts
1. Use `pdftotext -layout` (preserves columns better than default)
2. Or use PyMuPDF with block extraction and manual column detection
3. Verify by rendering page to image

### For Images/Figures
1. Use `pdfimages -list` to list all images
2. Use `pdfimages -png` to extract all images
3. Use `pdftoppm` to render full pages with figures in context

### For Cross-Paper Comparison
1. Extract tables from both PDFs using pdfplumber
2. Save to separate files for diffing
3. Render relevant pages to images for side-by-side comparison

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Table cells merged/lost | Use pdfplumber + verify with pdftoppm render |
| Column order wrong | Use `pdftotext -layout` or PyMuPDF block extraction |
| Missing content | Render page to image and inspect visually |
| Garbled text | Try PyMuPDF (better Unicode handling) |
| Large file | Extract specific pages: `pdftotext -f 5 -l 10` |

## Environment Notes
- Python interpreter: `/home/luwa/.conda/envs/dsclr/bin/python`
- All poppler tools available system-wide
- Temp files: use `/tmp/` for intermediate outputs
- For CUDA/ML tasks, see project_rules.md (use terminal directly, not sandbox)
