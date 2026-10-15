import pymupdf

doc = pymupdf.open("EY_Scanned_PDF_OCR_Practice.pdf")

page = doc[0]

pix = page.get_pixmap(dpi=300)
pix.save("scanned_page.png")

import pytesseract
from PIL import Image

image = Image.open("scanned_page.png")

text = pytesseract.image_to_string(image)

print(text)

from pathlib import Path


print("Forklift FL-17" in text)


print("injuries" in text.lower())

# Extract lines mentioning inspection
inspection_lines = [
    line for line in text.splitlines()
    if "inspection" in line.lower()
]
print(inspection_lines)

# Count whole-word occurrences
inspection_count = text.lower().split().count("inspection")
print("Inspection count:", inspection_count)

# Print all lines containing Loading Bay
loading_bay_lines = [
    line for line in text.splitlines()
    if "loading bay" in line.lower()
]
print(loading_bay_lines)

# Save OCR output
Path("ocr_result.txt").write_text(
    text, encoding="utf-8"
)