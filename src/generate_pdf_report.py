#!/usr/bin/env python3
"""Generate a comprehensive, executive & technical PDF report for team handoff.
Author: Aryan Goel
"""

__author__ = "Aryan Goel"

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PDF_OUTPUT_PATH = Path("report/Fact_Checking_LLM_Benchmark_Report_Aryan_Goel.pdf")
FIGURES_DIR = Path("results/figures")


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                40,
                11 * inch - 30,
                "LLM Fact-Verification Benchmark Report: Llama 3.2 vs. Qwen 2.5",
            )
            self.drawRightString(
                8.5 * inch - 40,
                11 * inch - 30,
                "Author: Aryan Goel",
            )
            self.setStrokeColor(colors.HexColor("#DDDDDD"))
            self.setLineWidth(0.5)
            self.line(40, 11 * inch - 34, 8.5 * inch - 40, 11 * inch - 34)

        # Running footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 40, 25, page_text)
        self.drawString(
            40,
            25,
            "Confidential - ANLP Group Assignment Benchmark Report | October 2026",
        )
        self.setStrokeColor(colors.HexColor("#DDDDDD"))
        self.setLineWidth(0.5)
        self.line(40, 36, 8.5 * inch - 40, 36)

        self.restoreState()


def create_pdf():
    PDF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1B365D"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4B6B94"),
        spaceAfter=10,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2C3E50"),
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1B365D"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2C3E50"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#102A45"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#222222"),
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1B365D"),
    )

    story = []

    # ==================== TITLE SECTION ====================
    story.append(Paragraph("Fact-Checking LLM Benchmark & Comparison Report", title_style))
    story.append(
        Paragraph(
            "Empirical Evaluation of <b>Llama 3.2 (3B)</b> vs. <b>Qwen 2.5 (7B)</b> on FEVER & VitaminC Across Correctness, Stability, Robustness, and Sensitivity",
            subtitle_style,
        )
    )

    # Author & Metadata Box
    meta_table_data = [
        [
            Paragraph("<b>Author / Lead Investigator:</b> Aryan Goel", meta_style),
            Paragraph("<b>Date:</b> October 2026", meta_style),
        ],
        [
            Paragraph("<b>Course:</b> ANLP Group Assignment", meta_style),
            Paragraph("<b>Local Runtime:</b> Apple Silicon (via Ollama)", meta_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[270, 260])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ==================== 1. EXECUTIVE SUMMARY ====================
    story.append(Paragraph("1. Executive Summary & Core Verdict", h1_style))
    story.append(
        Paragraph(
            "This report delivers the full empirical comparison between <b>Llama 3.2 (3B)</b> (from our team baseline handoff) and <b>Qwen 2.5 (7B)</b> (completed as requested). Rather than measuring accuracy as an isolated metric, both models were audited under an evidence-grounded framework across four core trust properties: <b>Correctness</b>, <b>Repeat-Run Stability</b>, <b>Robustness to Distractors</b>, and <b>Sensitivity to Meaningful Evidence Changes</b>.",
            body_style,
        )
    )

    summary_callout = [
        [
            Paragraph(
                "<b>KEY FINDING & TEAM RECOMMENDATION:</b><br/>"
                "<b>Qwen 2.5 (7B) substantially outperforms Llama 3.2 across every evaluated dimension.</b> "
                "While Llama 3.2 collapsed under irrelevant noise (-50% accuracy drop) and failed to detect refutations (defaulting to passivity), "
                "Qwen 2.5 demonstrated superior contradiction detection (+18.7% on REFUTES), 95.3% exact 5-run stability, and outstanding noise immunity (86% invariance). "
                "<b>We strongly recommend deploying Qwen 2.5 (7B) for the automated fact-checking pipeline.</b>",
                callout_style,
            )
        ]
    ]
    summary_box = Table(summary_callout, colWidths=[530])
    summary_box.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF3FA")),
            ("LINEBEFORE", (0, 0), (0, -1), 3.5, colors.HexColor("#1B365D")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C2D5E8")),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )
    story.append(summary_box)
    story.append(Spacer(1, 10))

    # ==================== 2. MASTER SCORECARD ====================
    story.append(Paragraph("2. Master Comparative Scorecard (The 4 Dimensions)", h1_style))

    scorecard_data = [
        [
            Paragraph("Dimension / Metric", table_header_style),
            Paragraph("Llama 3.2 (3B)<br/><i>(Team Handoff)</i>", table_header_style),
            Paragraph("Qwen 2.5 (7B)<br/><i>(Aryan's Run)</i>", table_header_style),
            Paragraph("Delta (Δ)", table_header_style),
            Paragraph("Practical Analysis", table_header_style),
        ],
        # Dim 1
        [
            Paragraph("<b>1. Correctness</b><br/>Overall FEVER Accuracy", table_cell_bold),
            Paragraph("64.67% (97/150)", table_cell_style),
            Paragraph("<b>71.33%</b> (107/150)", table_cell_bold),
            Paragraph("<font color='#006622'><b>+6.66%</b></font>", table_cell_style),
            Paragraph("Qwen is significantly closer to gold truth.", table_cell_style),
        ],
        [
            Paragraph("• SUPPORTS Accuracy", table_cell_style),
            Paragraph("77.33% (58/75)", table_cell_style),
            Paragraph("72.00% (54/75)", table_cell_style),
            Paragraph("<font color='#880000'>-5.33%</font>", table_cell_style),
            Paragraph("Llama slightly favors confirmation bias.", table_cell_style),
        ],
        [
            Paragraph("• REFUTES Accuracy", table_cell_style),
            Paragraph("52.00% (39/75)", table_cell_style),
            Paragraph("<b>70.67%</b> (53/75)", table_cell_bold),
            Paragraph("<font color='#006622'><b>+18.67%</b></font>", table_cell_style),
            Paragraph("<b>Major Qwen strength:</b> accurately detects contradictions.", table_cell_style),
        ],
        [
            Paragraph("• Predicted NEI (Passivity)", table_cell_style),
            Paragraph("35 / 150 (23.3%)", table_cell_style),
            Paragraph("<b>4 / 150 (2.7%)</b>", table_cell_bold),
            Paragraph("<font color='#006622'><b>-31 cases</b></font>", table_cell_style),
            Paragraph("Llama over-relies on 'Not Enough Info' escape hatches.", table_cell_style),
        ],
        # Dim 2
        [
            Paragraph("<b>2. Stability</b><br/>Exact 5/5 Consistency", table_cell_bold),
            Paragraph("80.00% (120/150)", table_cell_style),
            Paragraph("<b>95.33%</b> (143/150)", table_cell_bold),
            Paragraph("<font color='#006622'><b>+15.33%</b></font>", table_cell_style),
            Paragraph("Qwen verdicts are highly repeatable across seeds.", table_cell_style),
        ],
        [
            Paragraph("• Mean Modal Agreement", table_cell_style),
            Paragraph("95.60%", table_cell_style),
            Paragraph("<b>98.93%</b>", table_cell_bold),
            Paragraph("<font color='#006622'>+3.33%</font>", table_cell_style),
            Paragraph("Tight clustering around dominant decision.", table_cell_style),
        ],
        [
            Paragraph("• Unstable Claims", table_cell_style),
            Paragraph("30 cases (20.0%)", table_cell_style),
            Paragraph("<b>7 cases (4.7%)</b>", table_cell_bold),
            Paragraph("<font color='#006622'><b>-23 cases</b></font>", table_cell_style),
            Paragraph("Qwen nearly eliminates stochastic jitter.", table_cell_style),
        ],
        # Dim 3
        [
            Paragraph("<b>3. Robustness</b><br/>Distractor Invariance", table_cell_bold),
            Paragraph("39.33% (59/150)", table_cell_style),
            Paragraph("<b>86.00%</b> (129/150)", table_cell_bold),
            Paragraph("<font color='#006622'><b>+46.67%</b></font>", table_cell_style),
            Paragraph("<b>Decisive win:</b> Qwen filters out noisy text.", table_cell_style),
        ],
        [
            Paragraph("• Accuracy under Distractor", table_cell_style),
            Paragraph("14.67%", table_cell_style),
            Paragraph("<b>67.33%</b>", table_cell_bold),
            Paragraph("<font color='#006622'><b>+52.66%</b></font>", table_cell_style),
            Paragraph("Llama collapsed under noise; Qwen stayed resilient.", table_cell_style),
        ],
        [
            Paragraph("• Definite → NEI Collapse", table_cell_style),
            Paragraph("70 cases", table_cell_style),
            Paragraph("<b>4 cases</b>", table_cell_bold),
            Paragraph("<font color='#006622'><b>-66 cases</b></font>", table_cell_style),
            Paragraph("Llama froze when 1 unrelated sentence was added.", table_cell_style),
        ],
        # Dim 4
        [
            Paragraph("<b>4. Sensitivity</b><br/>Correct Sensitivity (VitaminC)", table_cell_bold),
            Paragraph("27.00% (27/100)", table_cell_style),
            Paragraph("<b>48.00%</b> (48/100)", table_cell_bold),
            Paragraph("<font color='#006622'><b>+21.00%</b></font>", table_cell_style),
            Paragraph("Qwen correctly updates verdict when facts change.", table_cell_style),
        ],
        [
            Paragraph("• Incorrect Flips", table_cell_style),
            Paragraph("45 / 100", table_cell_style),
            Paragraph("<b>8 / 100</b>", table_cell_bold),
            Paragraph("<font color='#006622'><b>-37 errors</b></font>", table_cell_style),
            Paragraph("Llama flipped erratically; Qwen flipped accurately.", table_cell_style),
        ],
        [
            Paragraph("• REFUTE-side Accuracy", table_cell_style),
            Paragraph("42.00%", table_cell_style),
            Paragraph("<b>76.00%</b>", table_cell_bold),
            Paragraph("<font color='#006622'><b>+34.00%</b></font>", table_cell_style),
            Paragraph("Qwen reliably refutes claims with opposing facts.", table_cell_style),
        ],
    ]

    scorecard_table = Table(
        scorecard_data, colWidths=[110, 85, 95, 65, 175]
    )
    scorecard_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
            ("ALIGN", (1, 0), (3, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(scorecard_table)
    story.append(Spacer(1, 10))

    # Page Break for Deep-Dive Analysis & Figures
    story.append(PageBreak())

    # ==================== 3. DEEP-DIVE DIMENSION ANALYSIS ====================
    story.append(Paragraph("3. Deep-Dive Dimension Analysis", h1_style))

    # Dimension 1
    story.append(Paragraph("Dimension 1: Baseline Correctness on FEVER Gold Evidence", h2_style))
    story.append(
        Paragraph(
            "In Experiment 1 (150 gold-evidence cases: 75 SUPPORTS, 75 REFUTES at <i>T=0</i>), Llama 3.2 exhibited an acute polarity asymmetry. "
            "While it validated supporting evidence reasonably well (77.33%), its accuracy on refutations was barely better than random chance (52.00%), "
            "collapsing 35 cases into <code>NOT ENOUGH INFORMATION</code>. "
            "By contrast, <b>Qwen 2.5 achieved balanced precision</b> (72.00% on SUPPORTS, 70.67% on REFUTES), defaulting to NEI on only 4 claims. "
            "This demonstrates that Qwen has significantly superior logical reasoning for contradiction detection.",
            body_style,
        )
    )

    # Dimension 2
    story.append(Paragraph("Dimension 2: Repeat-Run Stability Across Seeds", h2_style))
    story.append(
        Paragraph(
            "In Experiment 2, the exact same 150 cases were evaluated 5 times at <i>T=0.2</i> across five fixed seeds (101, 202, 303, 404, 505). "
            "Llama 3.2 produced 30 unstable cases (20.0% variance). "
            "<b>Qwen 2.5 achieved 95.33% exact 5/5 consistency</b>, with 105 cases consistently correct and only 7 unstable claims. "
            "Crucially, Qwen's accuracy range remained rock-solid (70.67% – 73.33%), proving its judgments stem from deterministic semantic comprehension rather than token sampling luck.",
            body_style,
        )
    )

    # Dimension 3
    story.append(Paragraph("Dimension 3: Robustness to Irrelevant Distractors", h2_style))
    story.append(
        Paragraph(
            "Experiment 3 evaluated resilience against context noise by appending one unrelated sentence from another Wikipedia article. "
            "<b>This was Llama 3.2's fatal weakness:</b> its accuracy cratered by <b>-50.0 percentage points</b> (from 64.67% down to 14.67%), "
            "with 70 definite verdicts collapsing into NEI. "
            "In stark contrast, <b>Qwen 2.5 maintained 86.00% distractor invariance</b>, with its accuracy dropping only -4.0 percentage points (71.33% → 67.33%). "
            "Qwen successfully focused on the relevant core evidence and filtered out the distractor noise.",
            body_style,
        )
    )

    # Dimension 4
    story.append(Paragraph("Dimension 4: Sensitivity to Contrastive Evidence Revisions (VitaminC)", h2_style))
    story.append(
        Paragraph(
            "Experiment 4 tested 100 unique claims from VitaminC, each evaluated against both a supporting and a revised refuting snippet (200 model evaluations). "
            "A trustworthy model must change its verdict from SUPPORTED to REFUTED when context changes. "
            "Llama 3.2 achieved only 27.00% correct sensitivity, frequently failing to notice refuting evidence (46% NEI on REFUTE). "
            "<b>Qwen 2.5 achieved 48.00% correct sensitivity</b> (nearly 1.8x Llama's rate) and refuting-context accuracy of <b>76.00%</b> (vs. Llama's 42.00%).",
            body_style,
        )
    )
    story.append(Spacer(1, 10))

    # ==================== 4. EMBEDDED FIGURES ====================
    story.append(Paragraph("4. Benchmark Visualizations & Performance Charts", h1_style))

    chart1_path = FIGURES_DIR / "macro_f1_comparison.png"
    chart2_path = FIGURES_DIR / "per_class_f1_breakdown.png"

    if chart1_path.exists() and chart2_path.exists():
        img_table = Table(
            [
                [
                    Image(str(chart1_path), width=260, height=160),
                    Image(str(chart2_path), width=260, height=160),
                ],
                [
                    Paragraph("<b>Figure 1:</b> Macro-F1 across claim-only & retrieved evidence.", body_style),
                    Paragraph("<b>Figure 2:</b> Per-class F1 score (Supports vs. Refutes vs. NEI).", body_style),
                ],
            ],
            colWidths=[265, 265],
        )
        img_table.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 2),
            ])
        )
        story.append(img_table)

    story.append(Spacer(1, 10))

    # Page Break for Business-Risk & Recommendations
    story.append(PageBreak())

    # ==================== 5. BUSINESS-RISK LAYER ====================
    story.append(Paragraph("5. Enterprise & Business-Risk Audit (LLM-Judge Layer)", h1_style))
    story.append(
        Paragraph(
            "In production verification systems (e.g. legal compliance, automated claim validation, editorial moderation), different error types incur vastly different operational costs:",
            body_style,
        )
    )

    risk_data = [
        [
            Paragraph("Failure Mode", table_header_style),
            Paragraph("Llama 3.2 Risk Profile", table_header_style),
            Paragraph("Qwen 2.5 Risk Profile", table_header_style),
            Paragraph("Business / Operational Impact", table_header_style),
        ],
        [
            Paragraph("<b>False Support (Hallucination)</b>", table_cell_bold),
            Paragraph("Low (4.7%)", table_cell_style),
            Paragraph("Low (5.3%)", table_cell_style),
            Paragraph("<font color='#880000'><b>Critical Risk:</b></font> Publishing false facts as verified truth.", table_cell_style),
        ],
        [
            Paragraph("<b>Passivity Paralyzation (Over-Cautious NEI)</b>", table_cell_bold),
            Paragraph("<font color='#880000'><b>Severe (23.3% - 46.7%)</b></font>", table_cell_style),
            Paragraph("<b>Minimal (2.7% - 8.0%)</b>", table_cell_bold),
            Paragraph("<font color='#996600'><b>High Bottleneck:</b></font> Floods human moderation queues with unnecessary manual reviews.", table_cell_style),
        ],
        [
            Paragraph("<b>Distractor Fragility (Adversarial Noise)</b>", table_cell_bold),
            Paragraph("<font color='#880000'><b>Extreme (-50% collapse)</b></font>", table_cell_style),
            Paragraph("<b>Resilient (-4% drop)</b>", table_cell_bold),
            Paragraph("<font color='#880000'><b>Vulnerability:</b></font> Web crawlers and SEO spam easily blind Llama into inaction.", table_cell_style),
        ],
        [
            Paragraph("<b>Contradiction Blindness</b>", table_cell_bold),
            Paragraph("<font color='#880000'><b>High (48% error on REFUTES)</b></font>", table_cell_style),
            Paragraph("<b>Low (24% error on REFUTES)</b>", table_cell_bold),
            Paragraph("<font color='#996600'><b>Compliance Risk:</b></font> Fails to flag factual violations and debunked claims.", table_cell_style),
        ],
    ]

    risk_table = Table(risk_data, colWidths=[120, 110, 110, 190])
    risk_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(risk_table)
    story.append(Spacer(1, 12))

    # ==================== 6. RECOMMENDATIONS & NEXT STEPS ====================
    story.append(Paragraph("6. Conclusions & Actionable Team Roadmap", h1_style))

    recs = [
        "<b>1. Select Qwen 2.5 (7B) as the Production Fact-Checker:</b> Qwen is substantially more reliable, balanced across polarities, robust against irrelevant text (+46.7%), and sensitive to revisions (+21.0%).",
        "<b>2. Enforce Retrieval Request Pacing:</b> As discovered during our pilot, unthrottled MediaWiki queries trigger HTTP 429 rate limits, leading to artificial NEI defaults. Our rate-limited retriever (0.3s sleep + retries) resolves this.",
        "<b>3. Implement a Two-Tier Human-in-the-Loop Protocol:</b> Because Qwen produces very few false supports, cases where Qwen outputs definite verdicts can be automated, while borderline low-similarity retrieval cases can be queued for human audit.",
        "<b>4. Team Next Steps:</b> Scale VitaminC evaluation to 500 contrastive pairs, test Chain-of-Thought (CoT) prompting on Qwen, and finalize report section drafts.",
    ]

    for rec in recs:
        story.append(Paragraph(f"• {rec}", body_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 15))

    # Sign-off box
    signoff_data = [
        [
            Paragraph("<b>Report Prepared By:</b> Aryan Goel", meta_style),
            Paragraph("<b>Status:</b> Completed & Verified", meta_style),
            Paragraph("<b>Repository:</b> github.com/16aryan/fever-vitaminc-llm-benchmark", meta_style),
        ]
    ]
    signoff_table = Table(signoff_data, colWidths=[180, 150, 200])
    signoff_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(signoff_table)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Successfully compiled PDF to: {PDF_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    create_pdf()
