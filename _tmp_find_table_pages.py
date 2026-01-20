from pdfminer.high_level import extract_text

pdf_path = 'kolafa1994 (1).pdf'
for page_num in range(0, 40):
    text = extract_text(pdf_path, page_numbers=[page_num]) or ''
    if 'TABLE' in text:
        print('page', page_num+1)
        # print a snippet around TABLE
        idx = text.find('TABLE')
        print(text[idx:idx+300].replace('\n',' '))
