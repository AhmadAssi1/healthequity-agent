"""Screen-readable PDF edition of the management report."""

import base64
import io
from pathlib import Path

from fpdf import FPDF, FPDF_VERSION
from matplotlib import get_data_path

INK = (32, 52, 64)
ACCENT = (21, 91, 109)
MUTED = (62, 86, 98)
RULE = (197, 215, 222)


class ManagementPDF(FPDF):
    def __init__(self, title, generated_at):
        super().__init__(unit="mm", format=(96, 176))
        self.set_margins(6.5, 14, 6.5)
        self.set_auto_page_break(True, margin=13)
        self.c_margin = 0
        fonts = Path(get_data_path()) / "fonts" / "ttf"
        self.add_font("Report", fname=fonts / "DejaVuSans.ttf")
        self.add_font("Report", style="B", fname=fonts / "DejaVuSans-Bold.ttf")
        self.set_title(title)
        self.set_author("HealthEquity Agent")
        self.set_producer(f"fpdf2 {FPDF_VERSION}")
        self.set_display_mode("fullpage", "continuous")
        self.section = "HealthEquity Agent"
        self.generated_at = generated_at

    def header(self):
        self.set_y(4)
        self.set_font("Report", "B", 9)
        self.set_text_color(*ACCENT)
        text = self.section
        while text and self.get_string_width(text) > self.epw - 4:
            text = text[:-1]
        self.cell(0, 5, text + ("…" if text != self.section else ""))
        self.set_draw_color(*RULE)
        self.line(self.l_margin, 11, self.w - self.r_margin, 11)
        self.set_xy(self.l_margin, self.t_margin)

    def footer(self):
        self.set_y(-9)
        self.set_font("Report", size=8.5)
        self.set_text_color(*MUTED)
        self.cell(self.epw * 0.7, 4, self.generated_at)
        self.cell(0, 4, f"Page {self.page_no()}", align="R")

    def paragraph(self, text, *, size=11, bold=False, color=INK, after=2):
        self.set_font("Report", "B" if bold else "", size)
        self.set_text_color(*color)
        self.multi_cell(0, self.font_size * 1.35, str(text), align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(after)

    def heading(self, text, size):
        self.set_font("Report", "B", size)
        height = self.multi_cell(0, self.font_size * 1.35, text, align="L",
                                 dry_run=True, output="HEIGHT")
        if self.will_page_break(height + 9):
            self.add_page()
        self.paragraph(text, size=size, bold=True, color=ACCENT)

    def field(self, label, value):
        # Inline text flows across the full page width, including long IDs or source labels.
        if self.will_page_break(10):
            self.add_page()
        self.set_text_color(*MUTED)
        self.set_font("Report", size=10)
        self.write(5, str(label) + "   ")
        self.set_text_color(*INK)
        self.set_font("Report", "B", 10.5)
        self.write(5, str(value))
        self.ln(6.3)

    def group_rates(self, fields):
        # Three short measurements fit across a phone. Keep each value with its label.
        width = self.epw / 3
        for start in range(0, len(fields), 3):
            batch = fields[start:start + 3]
            if self.will_page_break(13):
                self.add_page()
            for index, style, size, color in ((0, "", 10, MUTED), (1, "B", 10.5, INK)):
                self.set_font("Report", style, size)
                self.set_text_color(*color)
                top = bottom = self.y
                for column, pair in enumerate(batch):
                    self.set_xy(self.l_margin + column * width, top)
                    self.multi_cell(width - 1.5, 4.7, str(pair[index]), align="L",
                                    new_x="LMARGIN", new_y="NEXT")
                    bottom = max(bottom, self.y)
                self.set_xy(self.l_margin, bottom)
            self.ln(1.5)

    def records(self, block):
        headers, *rows = block["rows"]
        title_columns = 2 if block.get("role") == "headline" else 1
        for row in rows:
            if len(headers) == 2:
                self.field(*row)
                continue
            fields = list(zip(headers[title_columns:], row[title_columns:], strict=True))
            group_rates = block.get("role") == "group_rates"
            body_height = 11 * ((len(fields) + 2) // 3) if group_rates else 6.3 * len(fields)
            if self.will_page_break(12 + body_height):
                self.add_page()
            self.set_draw_color(*RULE)
            self.line(self.l_margin, self.y, self.w - self.r_margin, self.y)
            self.ln(2)
            self.heading(" / ".join(str(cell) for cell in row[:title_columns]), 12)
            if group_rates:
                self.group_rates(fields)
            else:
                for label, value in fields:
                    self.field(label, value)
            self.ln(1.5)

    def chart(self, block):
        self.set_font("Report", size=10)
        caption_height = self.multi_cell(0, self.font_size * 1.35, block["text"], align="L",
                                         dry_run=True, output="HEIGHT")
        height = self.epw * block["height_ratio"]
        if self.will_page_break(height + caption_height + 4):
            self.add_page()
        self.image(io.BytesIO(base64.b64decode(block["data"])), w=self.epw, h=height)
        self.ln(1)
        self.paragraph(block["text"], size=10, color=MUTED)


def write_compact_pdf(context, output):
    pdf = ManagementPDF(context["title"], context["generated_at"])
    for index, section in enumerate(context["sections"]):
        pdf.section = "HealthEquity Agent"
        pdf.add_page()
        pdf.section = section["title"]
        pdf.start_section(section["title"])
        if index == 0:
            pdf.paragraph(context["title"], size=18, bold=True, after=4)
        pdf.heading(section["title"], 15)
        for block in section["blocks"]:
            if block["kind"] == "table":
                pdf.records(block)
            elif block["kind"] == "chart":
                pdf.chart(block)
            elif block["kind"] == "heading":
                pdf.heading(block["text"], 12)
            else:
                prefix = "• " if block["kind"] == "bullet" else ""
                pdf.paragraph(prefix + block["text"])
    pdf.output(output)
