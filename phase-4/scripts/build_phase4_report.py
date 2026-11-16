import re, shutil, sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, KeepTogether, ListFlowable, ListItem


NAVY = colors.HexColor('#0F2A47'); TEAL = colors.HexColor('#1B7F8C'); GREY = colors.HexColor('#F1F4F7')
ss = getSampleStyleSheet()
body = ParagraphStyle('b', parent=ss['BodyText'], fontName='Helvetica', fontSize=9.6, leading=13.6, spaceAfter=4)
h1 = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=19, leading=23, textColor=NAVY, spaceAfter=8)
h2 = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12.2, leading=15, textColor=TEAL, spaceBefore=9, spaceAfter=4)
cell = ParagraphStyle('c', parent=body, fontSize=8.4, leading=11, spaceAfter=0)
cellh = ParagraphStyle('ch', parent=cell, fontName='Helvetica-Bold', textColor=colors.white)
code = ParagraphStyle('code', fontName='Courier', fontSize=8, leading=10.2, backColor=GREY, borderPadding=5, leftIndent=4, spaceAfter=6)

def inline(t):
    t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    t = re.sub(r'`([^`]+)`', r'<font name="Courier" size="8.4">\1</font>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(https?://[^\s<)]+)', r'<link href="\1" color="#1B7F8C">\1</link>', t)
    return t

def build(md_path, pdf_path, footer):
    lines = md_path.read_text(encoding='utf-8').splitlines()
    story, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            j = i + 1; blk = []
            while not lines[j].startswith('```'): blk.append(lines[j]); j += 1
            story += [Preformatted('\n'.join(blk), code)]; i = j + 1; continue
        if ln.startswith('# '): story += [Paragraph(inline(ln[2:]), h1)]
        elif ln.startswith('## '): story += [Paragraph(inline(ln[3:]), h2)]
        elif ln.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{2,}:?', c) for c in cells): rows.append(cells)
                i += 1
            n = max(len(r) for r in rows)
            data = [[Paragraph(inline(c), cellh if r == 0 else cell) for c in (row + [''] * (n - len(row)))] for r, row in enumerate(rows)]
            W = A4[0] - 36 * mm
            first = 0.34 if n > 2 else 0.4
            cw = [W * first] + [W * (1 - first) / (n - 1)] * (n - 1) if n > 1 else [W]
            if n >= 5: cw = [W * 0.22] + [W * 0.78 / (n - 1)] * (n - 1)
            t = Table(data, colWidths=cw, repeatRows=1)
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), NAVY), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, GREY]),
                                   ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#C9D2DB')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
            story += [t, Spacer(1, 6)]; continue
        elif re.match(r'^(- |\d+\. )', ln):
            items, ordered = [], bool(re.match(r'^\d+\. ', ln))
            while i < len(lines) and re.match(r'^(- |\d+\. )', lines[i]):
                items.append(ListItem(Paragraph(inline(re.sub(r'^(- |\d+\. )', '', lines[i])), body), leftIndent=14)); i += 1
            story += [ListFlowable(items, bulletType='1' if ordered else 'bullet', start='1' if ordered else None, leftIndent=14, bulletFontSize=8.5), Spacer(1, 3)]; continue
        elif ln.strip(): story += [Paragraph(inline(ln), body)]
        i += 1
    def foot(c, d):
        c.saveState(); c.setFont('Helvetica', 7.5); c.setFillColor(colors.grey)
        c.drawString(18 * mm, 10 * mm, footer); c.drawRightString(A4[0] - 18 * mm, 10 * mm, f'Page {d.page}'); c.restoreState()
    SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title=md_path.stem, author='Katarukonda Ashok').build(story, onFirstPage=foot, onLaterPages=foot)


from pathlib import Path
build(Path("reports/phase4/AIML_AshokKatarukonda_Phase4_Report.md"), Path("reports/phase4/AIML_AshokKatarukonda_Phase4_Report.pdf"), "Aura Workforce (CortexHR) | Katarukonda Ashok, GM00154 | Phase 4 report")
for f in ["Task16_Forecast", "Task17_Live_API", "Task18_Monitoring", "Task19_Console", "Task20_Model_Cards_Handover"]:
    pass
