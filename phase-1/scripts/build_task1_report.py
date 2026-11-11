from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

OUT = "Task1_Data_Audit/Task1_Documentation.pdf"
FIG = "Task1_Data_Audit/outputs/figures/"
BLUE = colors.HexColor("#1f4e79"); LIGHT = colors.HexColor("#e8f0f8")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.2, leading=12.2, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, spaceAfter=0)
cell = ParagraphStyle("cell", parent=body, fontSize=8.2, leading=10.2, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=BLUE, spaceAfter=2, alignment=0)
sub = ParagraphStyle("sub", parent=body, textColor=colors.HexColor("#555555"), fontSize=8.5, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=7, spaceAfter=3)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=1, spaceAfter=2)

def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bullet, bulletText="\u2022")
def tbl(rows, widths, header=True):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c4d3")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header: st += [("BACKGROUND", (0, 0), (-1, 0), BLUE)]
    for i in range(1, len(rows)):
        if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st)); return t
def img(name, w):
    from PIL import Image as PI
    iw, ih = PI.open(FIG + name).size
    return Image(FIG + name, width=w, height=w * ih / iw)

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
    c.drawString(18*mm, 10*mm, "Task 1 - Data Audit | Aura Workforce / CortexHR attrition | Katarukonda Ashok (GM00154)")
    c.drawRightString(A4[0]-18*mm, 10*mm, f"Page {d.page}"); c.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=15*mm, bottomMargin=16*mm,
                        title="Task 1 - Data Audit Report", author="Katarukonda Ashok")
W = A4[0] - 36*mm
s = []

s += [P("Task 1: Data Audit Report", h1),
      P("Employee attrition dataset (CortexHR / Aura Workforce) &nbsp;|&nbsp; Prepared by Katarukonda Ashok (GM00154) &nbsp;|&nbsp; October 2026", sub)]

s += [P("1. In short", h2),
      P("I checked the attrition table before building any model. It is <b>usable but not ready</b>: it needs a small clean-up, "
        "five columns must be removed because they reveal the answer, and the model must be judged with the right metrics because "
        "only about 1 in 7 employees leaves. Every problem below has a written decision, and those decisions become the rules of "
        "the Task 2 pipeline.")]

s += [tbl([
    ["What I checked", "What I found"],
    ["Size", "2,389 rows and 42 columns, one row per employee (snapshot 30 Sep 2025 plus a historical extension from 2021-2025). 6 rows were exact copies, leaving <b>2,383 employees</b>."],
    ["Missing values", "11 columns have gaps. Most are not data loss: they mean \"not applicable\" (never promoted, no raise yet, not reviewed yet)."],
    ["Outliers", "Two methods (IQR and z-score) disagree a lot. Only a handful of rows are real errors; most flagged rows are genuine senior staff or overtime workers."],
    ["Errors", "3 salaries 10x too high, 4 negative tenures, 2 attendance rates above 100%, 13 inconsistent city spellings, 10 negative experience values, 90 rows where tenure is longer than total experience."],
    ["Imbalance", "Only <b>14.6%</b> left (349 people). A model that always says \"stays\" would be 85.4% accurate and useless."],
    ["Leakage", "<b>6 columns</b> reveal the answer (exit date, resignation date, exit reason, exit interview, settlement, and the voluntary-exit flag)."],
], [30*mm, W - 30*mm])]

s += [KeepTogether([P("2. Missing values", h2),
      P("I first asked <i>why</i> each value is missing, because the right fix depends on the reason."),
      tbl([
    ["Column", "Missing", "Why it is missing", "Decision"],
    ["months_since_last_promotion", "61.0%", "Never promoted (explains 100% of gaps)", "Fill + add \"never promoted\" flag. Keep the column"],
    ["last_engagement_score", "22.2%", "Did not take the survey; these people leave slightly more (16.7% vs 14.1%)", "Add \"missing\" flag, then median fill"],
    ["last_raise_pct", "8.4%", "No raise received yet (100% of gaps)", "Fill 0 + flag"],
    ["last_performance_rating", "7.6%", "Recent joiners not yet reviewed (average tenure about 5 months)", "Flag, then median fill"],
    ["commute_distance_km", "2.3%", "Random gaps", "Median fill by city"],
    ["highest_education", "0.6%", "Random gaps", "Category \"Unknown\""],
    ["5 exit columns", "85-87%", "Blank by design for people who stayed", "Drop (leakage, section 5)"],
], [47*mm, 14*mm, 58*mm, W - 119*mm])])]
s += [Spacer(1, 3), P("<b>If a column were 40% missing, what would I do?</b> I would not drop it straight away. I would find out why it is blank. "
        "If the blank itself means something (like \"never promoted\"), I fill it sensibly and add a \"was missing\" flag so the signal is kept. "
        "I would drop it only if it is mostly empty and unrelated to leaving, or if it leaks the answer.")]
s += [img("missing_values.png", 92*mm)]

s += [KeepTogether([P("3. Outliers: IQR versus z-score", h2),
      P("An outlier is an unusual value. It is only a problem if it is an <i>error</i>. I compared two methods on the key numeric columns."),
      tbl([
    ["Column", "IQR", "Z-score", "What it really is"],
    ["annual_fixed_pay", "93", "26", "Right-skewed. Almost all are senior staff. On a log scale or within a job level only 1-9 remain. <b>3 are real typos</b>"],
    ["avg_overtime_hours_per_month_3m", "445", "61", "81% have zero overtime, so IQR flags every overtime worker. False alarm"],
    ["avg_hours_worked_3m", "48", "15", "One extreme record (EMP-0313: 1.8 hours, 20% attendance). Flag for HR"],
    ["tenure_months", "73", "45", "Long tenures up to 164 months are genuine. 4 negative values are errors"],
], [58*mm, 12*mm, 14*mm, W - 84*mm])])]
s += [Spacer(1, 3), P("<b>Lesson:</b> IQR assumes a roughly symmetric spread, so it over-flags skewed data. Z-score is less sensitive but is "
        "pulled by the outliers themselves. Neither method says \"this is an error\"; a business rule does. "
        "I therefore <b>keep genuine high values</b> and use <i>log(pay)</i> and <i>comp_ratio</i>, and I <b>fix only values that are impossible</b>.")]
s += [img("salary_outliers.png", 140*mm)]

s += [KeepTogether([P("4. Errors found and how I will fix them", h2),
      tbl([
    ["Problem", "Rows", "Fix"],
    ["Salary about 10x too high (EMP-0034, EMP-0433, EMP-0601); no longer matches comp_ratio", "3", "Divide by 10 and confirm with the Python team"],
    ["Negative tenure (join date after observation date)", "4", "Set to 0 and flag for source check"],
    ["Attendance rate 1.12 (above 100%)", "2", "Clip to 1.0"],
    ["City spelled differently (BENGALURU, Bangalore, Madras, delhi ncr...)", "13", "Standardise to one spelling"],
    ["wfh_share outside 0-1 (small noise on Onsite and Remote staff)", "213", "Clip to the range 0-1"],
    ["Negative total experience (historical rows)", "10", "Clip at 0"],
    ["Tenure more than 6 months longer than total experience", "90", "Raise experience to match tenure"],
    ["Exact duplicate employees", "6", "Drop the copies"],
], [W - 78*mm, 12*mm, 66*mm])])]

s += [KeepTogether([P("5. Class imbalance", h2),
      P("Of 2,383 employees, <b>349 left (14.6%)</b> and 2,034 stayed, about <b>5.8 stayers for every leaver</b>. "
        "Attrition also varies by department: D06 is highest at 21% and D10 lowest at 6.5%."),
      B("Accuracy is misleading. \"Nobody leaves\" is 85.4% accurate yet finds zero leavers.")])]
s += [B("I will judge models with <b>recall, precision, F1, ROC-AUC</b> (and PR-AUC), using stratified splits and class weights."),
      B("For HR, <b>recall</b> matters most: missing a leaver costs a resignation, while a false alarm only costs a conversation. Precision still has to stay reasonable so HR keeps trusting the list.")]
s += [img("class_balance.png", 140*mm)]

s += [KeepTogether([P("6. Data leakage: what must not reach the model", h2),
      P("Leakage means a column contains information that would not exist at the time we need to predict. "
        "Like studying with the answer key, the model looks perfect but fails in real life."),
      tbl([
    ["Column", "Why it leaks", "Decision"],
    ["exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount", "Filled for 92-100% of leavers and 0% of stayers. Created at exit", "Drop"],
    ["voluntary_left_within_12m", "A subset of the target itself", "Drop"],
    ["row_id, employee_id, source", "Identifiers and how the row was created, not real signal", "Exclude"],
    ["dept_attrition_rate_prior_12m, comp_ratio, attendance windows", "Measured on or before the snapshot date", "Safe to use"],
], [64*mm, 78*mm, W - 142*mm])])]
s += [Spacer(1, 3), P("<b>Proof:</b> a simple model on clean features scores ROC-AUC <b>0.71</b>. Adding just two exit columns lifts it to <b>1.00</b>. "
        "A perfect score is a warning sign, not a success. The strongest single clean feature (engagement score) reaches only 0.61, so a realistic "
        "final model should land near 0.70-0.75.")]
s += [img("leakage_demo.png", 70*mm)]

s += [P("7. Next steps (Task 2 pipeline rules)", h2)]
s += [B("Fit imputers, scalers and encoders on <b>training data only</b>, never on the whole dataset."),
      B("Dates span 2021-2025, so test a <b>time-based split</b> as well as stratified random CV."),
      B("Do not use gender or age (separate sheet) without a fairness review."),
      B("Open items for the team: confirm the 3 salary corrections, the 4 negative tenures and EMP-0313 with the Python team.")]

doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("built")
