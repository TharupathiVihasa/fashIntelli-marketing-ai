from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors

# Generate structured PDF reports
@dataclass
class ReportSection:
    heading: str
    paragraphs: List[str]
    table: Optional[Tuple[List[str], List[List[str]]]] = None
    images: Optional[List[Path]] = None


class PDFReportBuilder:
    def __init__(self, title: str) -> None:
        self.title = title
        self.styles = getSampleStyleSheet()

    def build(self, sections: Sequence[ReportSection], out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        story = []

        story.append(Paragraph(self.title, self.styles["Title"]))
        story.append(Spacer(1, 0.4 * cm))

        for sec in sections:
            story.append(Paragraph(sec.heading, self.styles["Heading2"]))
            story.append(Spacer(1, 0.2 * cm))

            for p in sec.paragraphs:
                story.append(Paragraph(p, self.styles["BodyText"]))
                story.append(Spacer(1, 0.15 * cm))

            if sec.table is not None:
                headers, rows = sec.table
                data = [headers] + rows
                table = Table(data, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9CA3AF")),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.3 * cm))

            if sec.images:
                for img_path in sec.images:
                    if img_path.exists():
                        story.append(Image(str(img_path), width=16 * cm, height=9 * cm))
                        story.append(Spacer(1, 0.2 * cm))

            story.append(Spacer(1, 0.3 * cm))

        doc.build(story)
        return out_path
