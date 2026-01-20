from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar

page_num = 8
rows = []
for page_layout in extract_pages('kolafa1994 (1).pdf', page_numbers=[page_num]):
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            for line in element:
                if isinstance(line, LTTextLine):
                    text = line.get_text().strip()
                    if text:
                        rows.append((line.y0, line.x0, text))

rows.sort(key=lambda r: (-r[0], r[1]))

groups = []
current = []
current_y = None
for y, x, text in rows:
    if current_y is None or abs(y - current_y) < 2.0:
        current.append((x, text))
        current_y = y if current_y is None else current_y
    else:
        groups.append(current)
        current = [(x, text)]
        current_y = y
if current:
    groups.append(current)

for group in groups:
    line = ' | '.join(text for x, text in sorted(group, key=lambda t: t[0]))
    if any(ch.isdigit() for ch in line):
        print(line)
