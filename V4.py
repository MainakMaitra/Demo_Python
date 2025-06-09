import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
import datetime

def create_precision_analysis_presentation():
    """Create a comprehensive PowerPoint presentation for Precision Drop Analysis"""
    
    # Create presentation object
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Define color scheme
    DARK_BLUE = RGBColor(30, 60, 114)  # #1e3c72
    MEDIUM_BLUE = RGBColor(42, 82, 152)  # #2a5298
    LIGHT_BLUE = RGBColor(232, 244, 248)  # #e8f4f8
    RED = RGBColor(255, 107, 107)  # #ff6b6b
    ORANGE = RGBColor(255, 167, 38)  # #ffa726
    GREEN = RGBColor(102, 187, 106)  # #66bb6a
    GRAY = RGBColor(102, 102, 102)  # #666666
    
    # Slide 1: Title Slide
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Add gradient background
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = DARK_BLUE
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(9), Inches(2))
    title_frame = title_box.text_frame
    title_frame.text = "Precision Drop Analysis"
    title_frame.paragraphs[0].font.size = Pt(48)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    title_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    # Subtitle
    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.5), Inches(9), Inches(1))
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.text = "Critical Insights & Action Plan"
    subtitle_frame.paragraphs[0].font.size = Pt(28)
    subtitle_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    subtitle_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    # Date
    date_box = slide.shapes.add_textbox(Inches(0.5), Inches(5), Inches(9), Inches(1))
    date_frame = date_box.text_frame
    date_frame.text = "Executive Presentation"
    date_frame.paragraphs[0].font.size = Pt(18)
    date_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    date_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    # Slide 2: Executive Summary
    slide_layout = prs.slide_layouts[5]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_frame.text = "Executive Summary"
    title_frame.paragraphs[0].font.size = Pt(36)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    
    # Add metrics cards
    metrics = [
        ("74.8%", "Current Precision", "4.8% Above Target", GREEN),
        ("96.7%", "FPs from Context Issues", "Critical", RED),
        ("$2.4M", "Annual Cost Impact", "At Risk", ORANGE),
        ("37/158", "Categories Below Target", "23.4%", RED)
    ]
    
    x_positions = [0.5, 2.75, 5, 7.25]
    for i, (value, label, status, color) in enumerate(metrics):
        # Card background
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x_positions[i]), Inches(1.5),
            Inches(2), Inches(2)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(255, 255, 255)
        card.line.color.rgb = RGBColor(224, 224, 224)
        card.line.width = Pt(1)
        
        # Value
        value_box = slide.shapes.add_textbox(
            Inches(x_positions[i]), Inches(1.7),
            Inches(2), Inches(0.6)
        )
        value_frame = value_box.text_frame
        value_frame.text = value
        value_frame.paragraphs[0].font.size = Pt(32)
        value_frame.paragraphs[0].font.bold = True
        value_frame.paragraphs[0].font.color.rgb = MEDIUM_BLUE
        value_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        # Label
        label_box = slide.shapes.add_textbox(
            Inches(x_positions[i]), Inches(2.3),
            Inches(2), Inches(0.4)
        )
        label_frame = label_box.text_frame
        label_frame.text = label
        label_frame.paragraphs[0].font.size = Pt(12)
        label_frame.paragraphs[0].font.color.rgb = GRAY
        label_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        # Status
        status_box = slide.shapes.add_textbox(
            Inches(x_positions[i]), Inches(2.8),
            Inches(2), Inches(0.3)
        )
        status_frame = status_box.text_frame
        status_frame.text = status
        status_frame.paragraphs[0].font.size = Pt(10)
        status_frame.paragraphs[0].font.color.rgb = color
        status_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    # Critical finding box
    finding_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(4.2),
        Inches(9), Inches(2)
    )
    finding_box.fill.solid()
    finding_box.fill.fore_color.rgb = RGBColor(255, 243, 205)  # Light yellow
    finding_box.line.color.rgb = RED
    finding_box.line.width = Pt(3)
    
    # Critical label
    label_box = slide.shapes.add_textbox(Inches(0.8), Inches(4.4), Inches(2), Inches(0.3))
    label_frame = label_box.text_frame
    label_frame.text = "CRITICAL FINDING"
    label_frame.paragraphs[0].font.size = Pt(12)
    label_frame.paragraphs[0].font.bold = True
    label_frame.paragraphs[0].font.color.rgb = RED
    
    # Finding text
    finding_text = slide.shapes.add_textbox(Inches(0.8), Inches(4.8), Inches(8.4), Inches(1.2))
    finding_frame = finding_text.text_frame
    finding_frame.text = "96.7% of False Positives stem from context-insensitive rule processing"
    finding_frame.paragraphs[0].font.size = Pt(18)
    finding_frame.paragraphs[0].font.bold = True
    
    p = finding_frame.add_paragraph()
    p.text = "Systematic erosion across categories requires immediate technical intervention combined with operational standardization."
    p.font.size = Pt(14)
    
    # Slide 3: Performance Against Targets
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_frame.text = "Performance vs. Targets"
    title_frame.paragraphs[0].font.size = Pt(36)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    
    # Add table
    rows = 4
    cols = 5
    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(9)
    height = Inches(2)
    
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    # Set column widths
    table.columns[0].width = Inches(3)
    table.columns[1].width = Inches(1.5)
    table.columns[2].width = Inches(1.5)
    table.columns[3].width = Inches(1.5)
    table.columns[4].width = Inches(1.5)
    
    # Header row
    headers = ['Metric', 'Target', 'Current', 'Status', 'Gap']
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = MEDIUM_BLUE
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.font.color.rgb = RGBColor(255, 255, 255)
        paragraph.font.bold = True
        paragraph.font.size = Pt(14)
    
    # Data rows
    data = [
        ['Primary: Overall Precision', '≥ 70%', '74.8%', 'EXCEEDING', '+4.8%'],
        ['Secondary: All Categories', '≥ 60%', '37/158 below 70%', 'FAILING', '-23.4%'],
        ['Tertiary: Validation Agreement', '≥ 85%', '83.6%', 'BELOW', '-1.4%']
    ]
    
    status_colors = {
        'EXCEEDING': GREEN,
        'FAILING': RED,
        'BELOW': ORANGE
    }
    
    for i, row_data in enumerate(data):
        for j, value in enumerate(row_data):
            cell = table.cell(i+1, j)
            cell.text = value
            if j == 3:  # Status column
                cell.fill.solid()
                cell.fill.fore_color.rgb = status_colors.get(value, GRAY)
                cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
            cell.text_frame.paragraphs[0].font.size = Pt(12)
    
    # Add trend chart
    chart_data = CategoryChartData()
    chart_data.categories = ['Nov 2024', 'Dec 2024', 'Jan 2025', 'Feb 2025', 'Mar 2025']
    chart_data.add_series('Precision %', (80.3, 70.5, 75.2, 70.8, 71.9))
    
    x, y, cx, cy = Inches(0.5), Inches(4.2), Inches(9), Inches(2.5)
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE, x, y, cx, cy, chart_data
    ).chart
    
    chart.has_title = True
    chart.chart_title.text_frame.text = "Precision Trend Analysis"
    
    # Slide 4: Critical Priority #1
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_frame.text = "Critical Priority #1: Context-Insensitive Negation"
    title_frame.paragraphs[0].font.size = Pt(32)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    
    # Critical finding box
    finding_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(1.3),
        Inches(9), Inches(1.2)
    )
    finding_box.fill.solid()
    finding_box.fill.fore_color.rgb = RGBColor(255, 243, 205)
    finding_box.line.color.rgb = RED
    finding_box.line.width = Pt(3)
    
    # Critical text
    critical_text = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(8.4), Inches(0.8))
    critical_frame = critical_text.text_frame
    critical_frame.text = "CRITICAL - 96.7% OF ALL FPS"
    critical_frame.paragraphs[0].font.size = Pt(14)
    critical_frame.paragraphs[0].font.bold = True
    critical_frame.paragraphs[0].font.color.rgb = RED
    
    p = critical_frame.add_paragraph()
    p.text = "Rules trigger on complaint keywords without understanding negation context"
    p.font.size = Pt(16)
    
    # Evidence boxes
    evidence_data = [
        ("Evidence", [
            "True Positives: 11.789 negations/transcript",
            "False Positives: 6.233 negations/transcript",
            "Risk Factor: 0.529"
        ]),
        ("Financial Impact", [
            "Annual Cost: $2.4M",
            "Per FP Cost: $100",
            "Volume: 24,000 FPs/year"
        ])
    ]
    
    x_pos = [0.5, 5]
    for i, (title, items) in enumerate(evidence_data):
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x_pos[i]), Inches(2.8),
            Inches(4.3), Inches(1.8)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(255, 235, 235) if i == 0 else RGBColor(255, 235, 235)
        box.line.color.rgb = RED
        box.line.width = Pt(2)
        
        text_box = slide.shapes.add_textbox(
            Inches(x_pos[i] + 0.2), Inches(3),
            Inches(3.9), Inches(1.4)
        )
        text_frame = text_box.text_frame
        text_frame.text = title
        text_frame.paragraphs[0].font.bold = True
        text_frame.paragraphs[0].font.size = Pt(14)
        
        for item in items:
            p = text_frame.add_paragraph()
            p.text = item
            p.font.size = Pt(12)
    
    # Example pattern
    example_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(5),
        Inches(9), Inches(0.8)
    )
    example_box.fill.solid()
    example_box.fill.fore_color.rgb = RGBColor(244, 244, 244)
    example_box.line.color.rgb = MEDIUM_BLUE
    example_box.line.width = Pt(2)
    
    example_text = slide.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(8.4), Inches(0.6))
    example_frame = example_text.text_frame
    example_frame.text = 'Customer: "I\'m NOT complaining, but I\'d like to understand my bill..."'
    example_frame.paragraphs[0].font.name = 'Courier New'
    example_frame.paragraphs[0].font.size = Pt(11)
    p = example_frame.add_paragraph()
    p.text = "System: [FLAGGED AS COMPLAINT]"
    p.font.name = 'Courier New'
    p.font.size = Pt(11)
    p.font.color.rgb = RED
    
    # Immediate actions
    actions_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(6),
        Inches(9), Inches(1.2)
    )
    actions_box.fill.solid()
    actions_box.fill.fore_color.rgb = LIGHT_BLUE
    
    actions_text = slide.shapes.add_textbox(Inches(0.8), Inches(6.1), Inches(8.4), Inches(1))
    actions_frame = actions_text.text_frame
    actions_frame.text = "Immediate Actions:"
    actions_frame.paragraphs[0].font.bold = True
    actions_frame.paragraphs[0].font.size = Pt(14)
    
    actions = [
        "→ Implement universal negation template for all queries",
        "→ Add context window expansion (10-word radius)",
        "→ Deploy to top 5 worst-performing categories first"
    ]
    
    for action in actions:
        p = actions_frame.add_paragraph()
        p.text = action
        p.font.size = Pt(12)
    
    # Slide 5: Critical Priority #2
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_frame.text = "Critical Priority #2: Agent Explanation Contamination"
    title_frame.paragraphs[0].font.size = Pt(32)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    
    # Priority box
    priority_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(1.3),
        Inches(9), Inches(0.8)
    )
    priority_box.fill.solid()
    priority_box.fill.fore_color.rgb = RGBColor(255, 243, 205)
    priority_box.line.color.rgb = ORANGE
    priority_box.line.width = Pt(3)
    
    priority_text = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(0.6))
    priority_frame = priority_text.text_frame
    priority_frame.text = "HIGH PRIORITY - 58.5% OF FPS"
    priority_frame.paragraphs[0].font.size = Pt(14)
    priority_frame.paragraphs[0].font.bold = True
    priority_frame.paragraphs[0].font.color.rgb = ORANGE
    
    p = priority_frame.add_paragraph()
    p.text = "Agent hypothetical scenarios incorrectly trigger complaint detection"
    p.font.size = Pt(16)
    
    # Contamination table
    rows = 5
    cols = 3
    left = Inches(0.5)
    top = Inches(2.5)
    width = Inches(9)
    height = Inches(2)
    
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    # Headers
    headers = ['Category', 'Contamination Rate', 'Impact']
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = MEDIUM_BLUE
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.text_frame.paragraphs[0].font.bold = True
    
    # Data
    contamination_data = [
        ['Fees & Interest', '75.0%', 'Highest'],
        ['Billing Disputes', '66.2%', 'High'],
        ['Customer Relations', '54.9%', 'Medium'],
        ['Overall Average', '58.5%', 'High']
    ]
    
    for i, row_data in enumerate(contamination_data):
        for j, value in enumerate(row_data):
            cell = table.cell(i+1, j)
            cell.text = value
            if j == 2:  # Impact column
                if value == 'Highest' or value == 'High':
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(255, 235, 235)
    
    # Example
    example_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(5),
        Inches(9), Inches(0.8)
    )
    example_box.fill.solid()
    example_box.fill.fore_color.rgb = RGBColor(244, 244, 244)
    
    example_text = slide.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(8.4), Inches(0.6))
    example_frame = example_text.text_frame
    example_frame.text = 'Agent: "If you were to complain about fees, you would need to..."'
    example_frame.paragraphs[0].font.name = 'Courier New'
    example_frame.paragraphs[0].font.size = Pt(11)
    p = example_frame.add_paragraph()
    p.text = "System: [FLAGGED AS COMPLAINT]"
    p.font.name = 'Courier New'
    p.font.size = Pt(11)
    p.font.color.rgb = RED
    
    # Solution approach
    solution_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(6),
        Inches(9), Inches(1.2)
    )
    solution_box.fill.solid()
    solution_box.fill.fore_color.rgb = LIGHT_BLUE
    
    solution_text = slide.shapes.add_textbox(Inches(0.8), Inches(6.1), Inches(8.4), Inches(1))
    solution_frame = solution_text.text_frame
    solution_frame.text = "Solution Approach:"
    solution_frame.paragraphs[0].font.bold = True
    
    solutions = [
        "→ Implement speaker role detection",
        "→ Add agent explanation filters",
        "→ Focus rules on customer channel only",
        "→ Expected Impact: +8-12% precision improvement"
    ]
    
    for solution in solutions:
        p = solution_frame.add_paragraph()
        p.text = solution
        p.font.size = Pt(12)
    
    # Slide 6: Emergency Categories
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_frame.text = "Emergency Category Review"
    title_frame.paragraphs[0].font.size = Pt(36)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    
    # Urgent box
    urgent_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(1.3),
        Inches(9), Inches(0.8)
    )
    urgent_box.fill.solid()
    urgent_box.fill.fore_color.rgb = RGBColor(255, 243, 205)
    urgent_box.line.color.rgb = RED
    urgent_box.line.width = Pt(3)
    
    urgent_text = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(0.6))
    urgent_frame = urgent_text.text_frame
    urgent_frame.text = "URGENT ACTION REQUIRED"
    urgent_frame.paragraphs[0].font.size = Pt(14)
    urgent_frame.paragraphs[0].font.bold = True
    urgent_frame.paragraphs[0].font.color.rgb = RED
    
    p = urgent_frame.add_paragraph()
    p.text = "3 Categories in Critical State with Severe MoM Degradation"
    p.font.size = Pt(16)
    
    # Emergency categories table
    rows = 4
    cols = 5
    left = Inches(0.5)
    top = Inches(2.5)
    width = Inches(9)
    height = Inches(2)
    
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    # Headers
    headers = ['Category', 'Current Precision', 'MoM Drop', 'Impact Score', 'Priority']
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = MEDIUM_BLUE
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(12)
    
    # Data
    emergency_data = [
        ['Credit Card ATM - Unclassified', '39.6%', '-', '14.6', 'EMERGENCY'],
        ['Credit Card ATM - Rejected/Declined', '47.3%', '-88.2%', '12.5', 'CRITICAL'],
        ['Customer Relations - Close Account', '56.2%', '-38.6%', '22.4', 'CRITICAL']
    ]
    
    for i, row_data in enumerate(emergency_data):
        for j, value in enumerate(row_data):
            cell = table.cell(i+1, j)
            cell.text = value
            if j == 4:  # Priority column
                cell.fill.solid()
                cell.fill.fore_color.rgb = RED
                cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
            cell.text_frame.paragraphs[0].font.size = Pt(11)
    
    # Category-specific actions
    actions_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(5),
        Inches(9), Inches(2)
    )
    actions_box.fill.solid()
    actions_box.fill.fore_color.rgb = LIGHT_BLUE
    
    actions_text = slide.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(8.4), Inches(1.8))
    actions_frame = actions_text.text_frame
    actions_frame.text = "Category-Specific Actions:"
    actions_frame.paragraphs[0].font.bold = True
    actions_frame.paragraphs[0].font.size = Pt(14)
    
    category_actions = [
        "→ Close Account: Add context filters for retention vs complaint",
        "→ Unclassified: Improve classification specificity",
        "→ Rejected/Declined: Separate technical issues from complaints",
        "→ Expected Impact: +25-30% precision for these categories"
    ]
    
    for action in category_actions:
        p = actions_frame.add_paragraph()
        p.text = action
        p.font.size = Pt(12)
    
    # Slide 7: Implementation Timeline
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_frame.text = "Implementation Timeline"
    title_frame.paragraphs[0].font.size = Pt(36)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    
    # Timeline items
    timeline_data = [
        ("Week 1-2", "Critical Fixes", [
            "Deploy universal negation template",
            "Fix top 3 emergency categories",
            "Implement daily monitoring dashboard"
        ]),
        ("Month 1", "Systematic Improvements", [
            "Query optimization program",
            "Enhanced validation process",
            "Pattern-based improvements"
        ]),
        ("Quarter 1", "Strategic Initiatives", [
            "ML-based FP prediction",
            "Context-aware rule engine",
            "Semantic understanding layer"
        ])
    ]
    
    y_pos = 1.5
    for period, title, items in timeline_data:
        # Timeline marker
        marker = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.5), Inches(y_pos),
            Inches(1.5), Inches(0.5)
        )
        marker.fill.solid()
        marker.fill.fore_color.rgb = MEDIUM_BLUE
        
        marker_text = slide.shapes.add_textbox(Inches(0.5), Inches(y_pos), Inches(1.5), Inches(0.5))
        marker_frame = marker_text.text_frame
        marker_frame.text = period
        marker_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        marker_frame.paragraphs[0].font.bold = True
        marker_frame.paragraphs[0].font.size = Pt(12)
        marker_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        # Content box
        content_box = slide.shapes.add_textbox(Inches(2.2), Inches(y_pos), Inches(7.3), Inches(1.5))
        content_frame = content_box.text_frame
        content_frame.text = title
        content_frame.paragraphs[0].font.bold = True
        content_frame.paragraphs[0].font.size = Pt(14)
        
        for item in items:
            p = content_frame.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(11)
        
        y_pos += 1.8
    
    # Expected outcomes boxes
    outcome_data = [
        ("Expected Outcomes", [
            "Current: 74.8%",
            "Expected Gain: +10-15%",
            "Target: 85-89%"
        ]),
        ("Investment Required", [
            "Resources: 3-5 FTEs",
            "Timeline: 12 weeks",
            "ROI: $2.4M saved annually"
        ])
    ]
    
    x_pos = [1, 5.5]
    for i, (title, items) in enumerate(outcome_data):
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x_pos[i]), Inches(5.5),
            Inches(3.5), Inches(1.5)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(255, 250, 240) if i == 0 else RGBColor(240, 248, 255)
        box.line.color.rgb = GREEN if i == 0 else MEDIUM_BLUE
        box.line.width = Pt(2)
        
        text_box = slide.shapes.add_textbox(
            Inches(x_pos[i] + 0.2), Inches(5.6),
            Inches(3.1), Inches(1.3)
        )
        text_frame = text_box.text_frame
        text_frame.text = title
        text_frame.paragraphs[0].font.bold = True
        text_frame.paragraphs[0].font.size = Pt(13)
        
        for item in items:
            p = text_frame.add_paragraph()
            p.text = item
            p.font.size = Pt(11)
    
    # Slide 8: Monitoring Framework
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_frame.text = "Monitoring & Risk Management"
    title_frame.paragraphs[0].font.size = Pt(36)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    
    # Risk indicators and alerts
    risk_data = [
        ("Critical Risk Indicators", [
            "Single-month precision drops >10%",
            "Validation agreement <75%",
            "New category launches without baseline",
            "Volume spikes >25% without adjustment"
        ]),
        ("Alert Thresholds", [
            "Precision drops >5% → Immediate alert",
            "Validation <80% → Weekly alert",
            "Volume spikes >20% → Operational alert",
            "Category <60% → Daily alert"
        ])
    ]
    
    x_pos = [0.5, 5]
    for i, (title, items) in enumerate(risk_data):
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x_pos[i]), Inches(1.5),
            Inches(4.3), Inches(2.2)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(255, 250, 250) if i == 0 else RGBColor(250, 250, 255)
        box.line.color.rgb = RED if i == 0 else ORANGE
        box.line.width = Pt(2)
        
        text_box = slide.shapes.add_textbox(
            Inches(x_pos[i] + 0.2), Inches(1.7),
            Inches(3.9), Inches(1.8)
        )
        text_frame = text_box.text_frame
        text_frame.text = title
        text_frame.paragraphs[0].font.bold = True
        text_frame.paragraphs[0].font.size = Pt(14)
        
        for item in items:
            p = text_frame.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(11)
    
    # Monitoring frequency table
    rows = 4
    cols = 3
    left = Inches(0.5)
    top = Inches(4)
    width = Inches(9)
    height = Inches(2.5)
    
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    # Headers
    headers = ['Frequency', 'Metrics', 'Actions']
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = MEDIUM_BLUE
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.text_frame.paragraphs[0].font.bold = True
    
    # Monitoring data
    monitoring_data = [
        ['Daily', 'Category precision, Volume', 'Real-time alerts, Anomaly detection'],
        ['Weekly', 'FP patterns, Operational metrics', 'Pattern analysis, Performance review'],
        ['Monthly', 'Validation effectiveness, Rule performance', 'Strategic assessment, Calibration']
    ]
    
    for i, row_data in enumerate(monitoring_data):
        for j, value in enumerate(row_data):
            cell = table.cell(i+1, j)
            cell.text = value
            cell.text_frame.paragraphs[0].font.size = Pt(11)
    
    # Slide 9: Key Recommendations
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_frame.text = "Key Recommendations & Next Steps"
    title_frame.paragraphs[0].font.size = Pt(36)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    
    # Recommendation boxes
    rec_data = [
        ("Immediate Actions (Week 1)", [
            "Deploy negation handling fix",
            "Emergency category intervention",
            "Validation team investigation"
        ], RED),
        ("Quick Wins (Month 1)", [
            "Agent contamination filters",
            "Query complexity reduction",
            "Monitoring dashboard launch"
        ], ORANGE),
        ("Strategic Goals (Quarter 1)", [
            "ML-based improvements",
            "Context-aware processing",
            "85%+ precision target"
        ], GREEN),
        ("Success Metrics", [
            "Overall precision: 85%+",
            "All categories: >60%",
            "Validation agreement: >85%",
            "$2.4M cost avoidance"
        ], MEDIUM_BLUE)
    ]
    
    positions = [(0.5, 1.5), (5, 1.5), (0.5, 4), (5, 4)]
    
    for i, (title, items, color) in enumerate(rec_data):
        x, y = positions[i]
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(y),
            Inches(4.3), Inches(2)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(255, 250, 250) if color == RED else RGBColor(255, 253, 240) if color == ORANGE else RGBColor(240, 255, 240) if color == GREEN else RGBColor(240, 248, 255)
        box.line.color.rgb = color
        box.line.width = Pt(2)
        
        text_box = slide.shapes.add_textbox(
            Inches(x + 0.2), Inches(y + 0.1),
            Inches(3.9), Inches(1.8)
        )
        text_frame = text_box.text_frame
        text_frame.text = title
        text_frame.paragraphs[0].font.bold = True
        text_frame.paragraphs[0].font.size = Pt(13)
        
        for item in items:
            p = text_frame.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(11)
    
    # Feasibility assessment
    feasibility_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(6.2),
        Inches(9), Inches(1)
    )
    feasibility_box.fill.solid()
    feasibility_box.fill.fore_color.rgb = RGBColor(212, 237, 218)  # Light green
    feasibility_box.line.color.rgb = GREEN
    feasibility_box.line.width = Pt(3)
    
    feasibility_text = slide.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(8.4), Inches(0.8))
    feasibility_frame = feasibility_text.text_frame
    feasibility_frame.text = "Feasibility Assessment"
    feasibility_frame.paragraphs[0].font.bold = True
    feasibility_frame.paragraphs[0].font.size = Pt(14)
    
    p = feasibility_frame.add_paragraph()
    p.text = "Implementation Success: Yes - achieving 85-89% precision target is feasible"
    p.font.size = Pt(12)
    
    p = feasibility_frame.add_paragraph()
    p.text = "Strategic Value: High - comprehensive framework addresses root causes while building sustainable improvement capabilities"
    p.font.size = Pt(12)
    
    # Save presentation
    prs.save('Precision_Drop_Analysis_Presentation.pptx')
    print("Presentation created successfully: Precision_Drop_Analysis_Presentation.pptx")
    print(f"Total slides: {len(prs.slides)}")
    print("\nSlide contents:")
    print("1. Title Slide")
    print("2. Executive Summary")
    print("3. Performance vs. Targets")
    print("4. Critical Priority #1: Context-Insensitive Negation")
    print("5. Critical Priority #2: Agent Explanation Contamination")
    print("6. Emergency Category Review")
    print("7. Implementation Timeline")
    print("8. Monitoring & Risk Management")
    print("9. Key Recommendations & Next Steps")

# Create the presentation
if __name__ == "__main__":
    create_precision_analysis_presentation()
