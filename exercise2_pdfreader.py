from pypdf import PdfReader
from pathlib import Path

reader = PdfReader("EY_Digital_PDF_Practice.pdf")

# 1. Number of pages
print("Pages:", len(reader.pages))

# 2. First page
page1_text = reader.pages[0].extract_text() or ""
print(page1_text)

# 3. Second page
page2_text = reader.pages[1].extract_text() or ""
print(page2_text)

# 4. Combine all pages
all_text = "\n".join(
    (page.extract_text() or "")
    for page in reader.pages
)

# 5. Search for a phrase
print("NorthStar Maintenance" in all_text)

# 6. Count whole-word occurrences of vendor
print("Vendor count:", all_text.lower().count("vendor"))

# 7. Save extracted content
Path("pypdf_output.txt").write_text(
    all_text, encoding="utf-8"
)
