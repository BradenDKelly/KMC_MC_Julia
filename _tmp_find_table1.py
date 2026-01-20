import pdfplumber
from pathlib import Path

pdf_path = Path('kolafa1994 (1).pdf')
with pdfplumber.open(str(pdf_path)) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ''
        if 'TABLE 1' in text:
            print('page', i+1)
            print(text)
            break
