from pdfminer.high_level import extract_text

pdf_path = 'kolafa1994 (1).pdf'
text = extract_text(pdf_path, page_numbers=[8])
print(text)
