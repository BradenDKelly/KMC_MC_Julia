from pathlib import Path
from pdfminer.high_level import extract_text

pdf_path = Path('kolafa1994 (1).pdf')
text = extract_text(str(pdf_path))
Path('_tmp_kolafa1994_text.txt').write_text(text, encoding='utf-8')
print('extracted', len(text))
