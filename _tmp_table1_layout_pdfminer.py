from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

laparams = LAParams(line_margin=0.2, char_margin=2.0, word_margin=0.1)
text = extract_text('kolafa1994 (1).pdf', page_numbers=[8], laparams=laparams)
print(text)
