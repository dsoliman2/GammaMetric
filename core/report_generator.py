"""
PDF Report Generator for LeapfrogDose
Generates professional PDF reports ready for Leapfrog submission.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from pathlib import Path


# Color scheme
DARK_BLUE = colors.HexColor("#2B5C8A")
LIGHT_BLUE = colors.HexColor("#D5E8F0")
ACCENT_GREEN = colors.HexColor("#2E8B57")
ACCENT_RED = colors.HexColor("#C0392B")
ACCENT_YELLOW = colors.HexColor("#D4A017")
GRAY = colors.HexColor("#666666")
LIGHT_GRAY = colors.HexColor("#F5F5F5")


def generate_charts(results: dict, df: pd.DataFrame, 
                    output_dir: str = ".") -> dict:
    """Generate analysis charts and save as images."""
    chart_paths = {}
    
    # --- Chart 1: Adult DLP by Region with Benchmarks ---
    if results["adult"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        regions = list(results["adult"].keys())
        fac_p50 = [results["adult"][r]["stats"]["p50"] for r in regions]
        fac_p25 = [results["adult"][r]["stats"]["p25"] for r in regions]
        fac_p75 = [results["adult"][r]["stats"]["p75"] for r in regions]
        
        x = np.arange(len(regions))
        width = 0.35
        
        # Facility bars
        bars = ax.bar(x, fac_p50, width, label='Facility Median (P50)',
                      color='#2B5C8A', alpha=0.85, zorder=3)
        
        # Error bars showing P25-P75 range
        ax.errorbar(x, fac_p50, 
                    yerr=[np.array(fac_p50) - np.array(fac_p25),
                          np.array(fac_p75) - np.array(fac_p50)],
                    fmt='none', color='#1a1a1a', capsize=5, zorder=4)
        
        # National benchmark markers
        from leapfrog_dose import ADULT_BENCHMARKS
        bench_p50 = []
        for r in regions:
            b = ADULT_BENCHMARKS.get(r, {})
            bench_p50.append(b.get("p50", 0))
        
        ax.scatter(x, bench_p50, color='#C0392B', marker='D', s=60,
                   zorder=5, label='National Median')
        
        ax.set_xlabel('')
        ax.set_ylabel('DLP (mGy·cm)')
        ax.set_title('Adult CT Dose by Body Region vs. National Benchmarks')
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace('-', '\n') for r in regions], fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        path = os.path.join(output_dir, "chart_adult_regions.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths["adult_regions"] = path
    
    # --- Chart 2: DLP Distribution Histogram for top region ---
    if results["adult"]:
        top_region = max(results["adult"].keys(), 
                        key=lambda r: results["adult"][r]["stats"]["n"])
        region_df = df[(df['body_region'] == top_region) & 
                       (df['age_group'] == 'Adult')]
        
        if len(region_df) > 10:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            
            ax.hist(region_df['dlp'], bins=30, color='#2B5C8A', 
                    alpha=0.7, edgecolor='white', zorder=3)
            
            stats = results["adult"][top_region]["stats"]
            ax.axvline(stats["p50"], color='#C0392B', linewidth=2,
                       linestyle='--', label=f'Facility P50: {stats["p50"]:.0f}')
            ax.axvline(stats["p75"], color='#D4A017', linewidth=2,
                       linestyle='--', label=f'Facility P75: {stats["p75"]:.0f}')
            
            bench = ADULT_BENCHMARKS.get(top_region, {})
            if bench:
                ax.axvline(bench["p50"], color='#2E8B57', linewidth=2,
                           linestyle=':', label=f'National P50: {bench["p50"]:.0f}')
            
            ax.set_xlabel('DLP (mGy·cm)')
            ax.set_ylabel('Number of Exams')
            ax.set_title(f'DLP Distribution: Adult {top_region} CT')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            path = os.path.join(output_dir, "chart_distribution.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths["distribution"] = path
    
    # --- Chart 3: Monthly Trend ---
    if 'exam_date' in df.columns and len(df) > 30:
        fig, ax = plt.subplots(figsize=(7, 3))
        
        monthly = df.groupby(df['exam_date'].dt.to_period('M'))['dlp'].median()
        monthly.index = monthly.index.to_timestamp()
        
        ax.plot(monthly.index, monthly.values, color='#2B5C8A', 
                linewidth=2, marker='o', markersize=4, zorder=3)
        ax.fill_between(monthly.index, monthly.values, alpha=0.1, 
                        color='#2B5C8A')
        
        ax.set_ylabel('Median DLP (mGy·cm)')
        ax.set_title('Monthly Median DLP Trend (All Exams)')
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        path = os.path.join(output_dir, "chart_trend.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths["trend"] = path
    
    return chart_paths


def generate_pdf_report(results: dict, df: pd.DataFrame,
                        output_path: str = "leapfrog_report.pdf"):
    """
    Generate a professional PDF report for Leapfrog submission.
    """
    output_dir = str(Path(output_path).parent)
    chart_paths = generate_charts(results, df, output_dir)
    
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.6*inch, bottomMargin=0.6*inch,
        leftMargin=0.75*inch, rightMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        'ReportTitle', parent=styles['Title'],
        fontSize=20, textColor=DARK_BLUE, spaceAfter=6,
        fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'SectionHeader', parent=styles['Heading2'],
        fontSize=13, textColor=DARK_BLUE, spaceBefore=18,
        spaceAfter=8, fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'SubHeader', parent=styles['Heading3'],
        fontSize=11, textColor=DARK_BLUE, spaceBefore=12,
        spaceAfter=6, fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'ReportBody', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor("#333333"),
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'SmallText', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=GRAY,
        fontName='Helvetica',
    ))
    
    story = []
    
    # ===== HEADER =====
    story.append(Paragraph(
        "CT Dose Analytics Report", styles['ReportTitle']))
    story.append(Paragraph(
        f"Leapfrog Survey Preparation", 
        ParagraphStyle('Subtitle', parent=styles['Normal'],
                       fontSize=12, textColor=GRAY, spaceAfter=4)))
    
    story.append(HRFlowable(
        width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))
    
    # Facility info table
    info_data = [
        ["Facility:", results["facility_name"],
         "Analysis Date:", results["analysis_date"][:10]],
        ["Reporting Period:", 
         f"{results['date_range']['start'][:10] if results['date_range']['start'] else 'N/A'}"
         f" to {results['date_range']['end'][:10] if results['date_range']['end'] else 'N/A'}",
         "Total Exams:", str(results["total_exams"])],
    ]
    
    info_table = Table(info_data, colWidths=[1.1*inch, 2.5*inch, 1.1*inch, 2.3*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), DARK_BLUE),
        ('TEXTCOLOR', (2, 0), (2, -1), DARK_BLUE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 12))
    
    # ===== SUMMARY BOX =====
    s = results["summary"]
    status_color = ACCENT_GREEN if s["regions_above_benchmark"] == 0 else ACCENT_RED
    
    summary_text = (
        f"<b>Summary:</b> Analyzed {s['total_adult_exams']} adult and "
        f"{s['total_peds_exams']} pediatric CT exams across "
        f"{s['regions_analyzed']} body regions. "
    )
    if s["regions_above_benchmark"] == 0:
        summary_text += "All regions within national benchmark levels."
    else:
        summary_text += (f"<font color='#C0392B'><b>{s['regions_above_benchmark']} "
                         f"region(s) above national benchmarks.</b></font>")
    
    if s["total_outliers"] > 0:
        summary_text += f" {s['total_outliers']} individual outlier exams flagged for review."
    
    story.append(Paragraph(summary_text, styles['ReportBody']))
    story.append(Spacer(1, 8))
    
    # ===== ADULT RESULTS TABLE =====
    story.append(Paragraph("Adult CT Dose Results", styles['SectionHeader']))
    story.append(Paragraph(
        "DLP values in mGy·cm. Status based on comparison to national "
        "ACR DIR benchmark percentiles.", styles['SmallText']))
    
    # Build table
    header = ['Body Region', 'N', 'P25', 'P50 (Median)', 'P75', 
              'Natl P50', 'Natl P75', 'Status']
    table_data = [header]
    
    from leapfrog_dose import ADULT_BENCHMARKS
    
    for region, data in results["adult"].items():
        st = data["stats"]
        bc = data["benchmark_comparison"]
        bench = bc.get("benchmarks", {}) or {}
        
        status = bc["status"]
        
        table_data.append([
            region, str(st["n"]),
            f'{st["p25"]:.0f}', f'{st["p50"]:.0f}', f'{st["p75"]:.0f}',
            f'{bench.get("p50", "—")}', f'{bench.get("p75", "—")}',
            status,
        ])
    
    t = Table(table_data, colWidths=[
        1.5*inch, 0.4*inch, 0.65*inch, 0.85*inch, 0.65*inch, 
        0.65*inch, 0.65*inch, 1.2*inch])
    
    t_style = [
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    
    # Color-code status column
    for i, row in enumerate(table_data[1:], start=1):
        status = row[-1]
        if status == "EXCELLENT":
            t_style.append(('TEXTCOLOR', (-1, i), (-1, i), ACCENT_GREEN))
            t_style.append(('FONTNAME', (-1, i), (-1, i), 'Helvetica-Bold'))
        elif status == "GOOD":
            t_style.append(('TEXTCOLOR', (-1, i), (-1, i), ACCENT_GREEN))
        elif status == "ABOVE BENCHMARK":
            t_style.append(('TEXTCOLOR', (-1, i), (-1, i), ACCENT_RED))
            t_style.append(('FONTNAME', (-1, i), (-1, i), 'Helvetica-Bold'))
    
    t.setStyle(TableStyle(t_style))
    story.append(t)
    
    # ===== CHART: Adult Regions =====
    if "adult_regions" in chart_paths:
        story.append(Spacer(1, 12))
        story.append(Image(chart_paths["adult_regions"], 
                          width=6.5*inch, height=3.25*inch))
    
    # ===== DISTRIBUTION CHART =====
    if "distribution" in chart_paths:
        story.append(Spacer(1, 8))
        story.append(Image(chart_paths["distribution"],
                          width=5.8*inch, height=2.9*inch))
    
    # ===== PEDIATRIC RESULTS =====
    if results["pediatric"]:
        story.append(PageBreak())
        story.append(Paragraph("Pediatric CT Dose Results", 
                              styles['SectionHeader']))
        story.append(Paragraph(
            "DLP values in mGy·cm by Leapfrog age strata. "
            "Minimum 10 encounters per stratum required for Leapfrog reporting.",
            styles['SmallText']))
        
        for region, age_data in results["pediatric"].items():
            story.append(Paragraph(f"{region}", styles['SubHeader']))
            
            header = ['Age Group', 'N', 'P25', 'P50', 'P75', 
                      'Natl P50', 'Natl P75', 'Status']
            table_data = [header]
            
            for age_grp, data in age_data.items():
                st = data["stats"]
                bc = data["benchmark_comparison"]
                bench = bc.get("benchmarks", {}) or {}
                
                table_data.append([
                    f"{age_grp} years", str(st["n"]),
                    f'{st["p25"]:.0f}', f'{st["p50"]:.0f}', f'{st["p75"]:.0f}',
                    f'{bench.get("p50", "—")}', f'{bench.get("p75", "—")}',
                    bc["status"],
                ])
            
            t = Table(table_data, colWidths=[
                1.0*inch, 0.4*inch, 0.65*inch, 0.65*inch, 0.65*inch,
                0.65*inch, 0.65*inch, 1.3*inch])
            
            t_style_peds = list(t_style)  # Copy base style
            
            for i, row in enumerate(table_data[1:], start=1):
                status = row[-1]
                if "ABOVE" in status:
                    t_style_peds.append(('TEXTCOLOR', (-1, i), (-1, i), ACCENT_RED))
                    t_style_peds.append(('FONTNAME', (-1, i), (-1, i), 'Helvetica-Bold'))
                elif status in ("EXCELLENT", "GOOD"):
                    t_style_peds.append(('TEXTCOLOR', (-1, i), (-1, i), ACCENT_GREEN))
            
            t.setStyle(TableStyle(t_style_peds))
            story.append(t)
            story.append(Spacer(1, 8))
    
    # ===== OUTLIERS =====
    if results["outliers"]:
        story.append(Paragraph("Flagged Outlier Exams", styles['SectionHeader']))
        story.append(Paragraph(
            "Exams with DLP exceeding 2× the facility's 75th percentile "
            "for that body region. These may indicate repeat scans, "
            "multi-phase studies, or protocol deviations warranting review.",
            styles['SmallText']))
        story.append(Spacer(1, 6))
        
        header = ['Region', 'DLP', 'Threshold', 'Date', 'Description']
        table_data = [header]
        
        for o in results["outliers"][:25]:
            table_data.append([
                o["region"],
                f'{o["dlp"]:.0f}',
                f'{o["threshold"]:.0f}' if o["threshold"] else "—",
                o["exam_date"][:10],
                o["study_description"][:35],
            ])
        
        t = Table(table_data, colWidths=[
            1.3*inch, 0.6*inch, 0.7*inch, 0.9*inch, 2.8*inch])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, 0), ACCENT_RED),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        
        if len(results["outliers"]) > 25:
            story.append(Paragraph(
                f"... and {len(results['outliers']) - 25} additional outliers. "
                f"Full list available in supplementary data.",
                styles['SmallText']))
    
    # ===== TREND CHART =====
    if "trend" in chart_paths:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Dose Trend Analysis", styles['SectionHeader']))
        story.append(Image(chart_paths["trend"],
                          width=5.8*inch, height=2.5*inch))
    
    # ===== FOOTER / METHODOLOGY =====
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Methodology:</b> DLP percentiles calculated from all qualifying "
        "CT examinations during the reporting period. Exams classified by "
        "body region using protocol/study description matching. Benchmarks "
        "based on ACR Dose Index Registry national reference levels. "
        "Outliers defined as exams with DLP > 2× facility 75th percentile. "
        "This report is prepared for Leapfrog Hospital Survey submission "
        "and internal dose optimization review.",
        styles['SmallText']))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"| LeapfrogDose Analytics",
        styles['SmallText']))
    
    # Build PDF
    doc.build(story)
    print(f"\n  PDF report saved: {output_path}")
    
    # Clean up chart images
    for path in chart_paths.values():
        try:
            os.remove(path)
        except Exception:
            pass
    
    return output_path
