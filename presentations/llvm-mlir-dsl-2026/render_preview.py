#!/usr/bin/env python3
"""Render the exported PDF and build a contact sheet. Requires PyMuPDF + Pillow."""
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
doc = fitz.open(HERE / "llvm-mlir-dsl-tutorial-v16.pdf")
preview = HERE / "preview"
preview.mkdir(exist_ok=True)
font = ImageFont.truetype(str(HERE / "assets" / "fonts" / "NVIDIASans_Rg.ttf"), 14)
thumbs = []
for i, page in enumerate(doc, 1):
    pix = page.get_pixmap(matrix=fitz.Matrix(1600 / page.rect.width, 1600 / page.rect.width), alpha=False)
    output = preview / f"slide-{i:02}.png"
    pix.save(output)
    thumb = Image.open(output).convert("RGB")
    thumb.thumbnail((400, 225))
    card = Image.new("RGB", (420, 255), "#eeeeee")
    card.paste(thumb, (10, 8))
    label = "Title" if i == 1 else f"{i - 1:02}"
    ImageDraw.Draw(card).text((12, 236), f"{label}  /  PDF {i:02}", font=font, fill="#616161")
    thumbs.append(card)
cols = 4
rows = (len(thumbs) + cols - 1) // cols
overview = Image.new("RGB", (cols * 420, rows * 255), "#eeeeee")
for i, card in enumerate(thumbs):
    overview.paste(card, ((i % cols) * 420, (i // cols) * 255))
overview.save(HERE / "overview.png")
print(f"Rendered {len(doc)} pages and overview.png")
