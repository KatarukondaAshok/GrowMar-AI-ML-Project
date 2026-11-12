"""Shared look and helpers for the task documentation PDFs."""
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BLUE = colors.HexColor("#1f4e79")
LIGHT = colors.HexColor("#e8f0f8")
W = A4[0] - 36 * mm
_ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=_ss["Normal"], fontName="Helvetica", fontSize=9.2, leading=12.2, spaceAfter=4)
cell = ParagraphStyle("cell", parent=body, fontSize=8.2, leading=10.2, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
h1 = ParagraphStyle("h1", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=BLUE, spaceAfter=2, alignment=0)
sub = ParagraphStyle("sub", parent=body, textColor=colors.HexColor("#555555"), fontSize=8.5, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=7, spaceAfter=3)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=1, spaceAfter=2)
code = ParagraphStyle("code", parent=body, fontName="Courier", fontSize=8, leading=10.5, backColor=colors.HexColor("#f2f2f2"),
                      leftIndent=4, rightIndent=4, borderPadding=3, spaceAfter=5)


def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bullet, bulletText="\u2022")
def C(t): return Paragraph(t.replace(" ", "&nbsp;").replace("\n", "<br/>"), code)


def tbl(rows, widths, header=True):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c4d3")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    for i in range(1, len(rows)):
        if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def img(path, width):
    iw, ih = PILImage.open(path).size
    return Image(path, width=width, height=width * ih / iw)


def keep(*flow): return KeepTogether(list(flow))


def build(out, story, title, footer_text):
    def footer(c, d):
        c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
        c.drawString(18 * mm, 10 * mm, footer_text)
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}"); c.restoreState()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=16 * mm, title=title, author="Katarukonda Ashok")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader
    print("built", out, "pages:", len(PdfReader(out).pages))


def checklist(items):
    """items: list of (criterion, status_text, evidence)."""
    rows = [["Completion criterion", "Status", "Evidence"]] + [[a, b, c] for a, b, c in items]
    return tbl(rows, [62 * mm, 16 * mm, W - 78 * mm])
