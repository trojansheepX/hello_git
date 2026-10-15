import pymupdf

doc = pymupdf.open("EY_Digital_PDF_Practice.pdf")

print("Pages:", len(doc))

# page = doc[0]
# text = page.get_text()
#
# print(text)

all_text = "\n".join(
    page.get_text() for page in doc
)

# if "maintenance" in all_text.lower():
#     print("Maintenance found")
# else:
#     print("Maintenance not found")

# count = all_text.lower().count("v303")
# print("V303 count:", count)

# page2_text = doc[1].get_text()
# print(page2_text)

from pathlib import Path

# Path("digital_pdf_text.txt").write_text(
#     all_text,
#     encoding="utf-8"
# )

# doc.close()