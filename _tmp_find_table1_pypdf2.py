from PyPDF2 import PdfReader

reader = PdfReader('kolafa1994 (1).pdf')
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ''
    if 'TABLE 1' in text:
        print('page', i+1)
        print(text)
        break
