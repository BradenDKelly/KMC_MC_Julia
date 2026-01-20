from pdfminer.high_level import extract_text

pdf_path = 'kolafa1994 (1).pdf'
# check pages for TABLE 1
for page_num in range(0, 40):
    text = extract_text(pdf_path, page_numbers=[page_num]) or ''
    if 'TABLE 1' in text:
        print('page', page_num+1)
        print(text)
        break
