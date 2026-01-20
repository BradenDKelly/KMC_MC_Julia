import re
from pathlib import Path
from pdfminer.high_level import extract_text

pdf_path = Path('1-s2.0-S1385894717320211-main.pdf')
text = extract_text(str(pdf_path))
Path('_tmp_pdf_text.txt').write_text(text, encoding='utf-8')

snippets = []
for keyword in ['kinetic', 'mobility', 'chemical potential', 'Rosenbluth', 'time', 'dt', 'mu', 'Rosenbluth scheme', 'mobility rate', 'chemical', 'Helmholtz']:
    m = re.search(r'.{0,500}%s.{0,500}' % re.escape(keyword), text, flags=re.IGNORECASE|re.DOTALL)
    if m:
        snippets.append(f'--- {keyword} ---\n' + m.group(0))

Path('_tmp_pdf_snippets.txt').write_text('\n\n'.join(snippets), encoding='utf-8')
print('snippets', len(snippets))
