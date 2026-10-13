# -*- coding: utf-8 -*-
"""
ECCE Customer-Experience Deep-Dive -- 3-slide deck (python-pptx).

Positioning: a CUSTOMER-EXPERIENCE exercise (find & explain bad experiences),
NOT an agent-scrutiny exercise. Operation / queue level throughout.

Palette: Synchrony brand -- graphite #323232 + Synchrony gold #FFB700.
Canvas : 13.333 x 7.5 in widescreen. All coordinates in inches (top-left origin).

No red/green in the brand palette, so better-vs-worse is encoded by
position + explicit labels + gold as the single "attention" accent
(gold underline on the higher-complaint number, gold "+NN pts" gap chip,
gold left-border on the WORSE quote). Numbers stay graphite for contrast;
gold is never used as body text on white (it fails contrast there).
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ------------------------------- palette -------------------------------
GRAPH = RGBColor(0x32, 0x32, 0x32)   # primary dark / graphite
GOLD  = RGBColor(0xFF, 0xB7, 0x00)   # Synchrony gold accent
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OFF   = RGBColor(0xF5, 0xF5, 0xF5)   # light card fill
MUTE  = RGBColor(0x6E, 0x6E, 0x6E)   # support text (on white)
LINE  = RGBColor(0xD8, 0xD8, 0xD8)   # card border
GCCC  = RGBColor(0xCC, 0xCC, 0xCC)   # support text (on graphite)
CARD  = RGBColor(0x3C, 0x3C, 0x3C)   # card fill on dark slides

TITLE_FONT = "Cambria"
BODY_FONT  = "Calibri"

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ------------------------------- helpers -------------------------------
def add_slide(bg=WHITE):
    slide = prs.slides.add_slide(BLANK)
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                               prs.slide_width, prs.slide_height)
    r.fill.solid(); r.fill.fore_color.rgb = bg
    r.line.fill.background(); r.shadow.inherit = False
    return slide


def add_rect(slide, x, y, w, h, fill, line=None, line_w=1.0, shape=MSO_SHAPE.RECTANGLE,
             radius=None):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE and radius is not None:
        # adjustment is a fraction of the shorter side (0..0.5)
        try:
            shp.adjustments[0] = max(0.0, min(0.5, radius / min(w, h)))
        except Exception:
            pass
    return shp


def _style_run(r, text, size, color, bold, font, italic, cs):
    r.text = text
    f = r.font
    f.size = Pt(size); f.bold = bold; f.italic = italic
    f.name = font; f.color.rgb = color
    if cs:  # character spacing in points -> centipoints attr
        r.font._rPr.set('spc', str(int(cs * 100)))


def _frame(tf, valign):
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = valign


def add_text(slide, x, y, w, h, text, size=12, color=GRAPH, bold=False,
             font=BODY_FONT, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
             italic=False, cs=0, line_spacing=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; _frame(tf, valign)
    p = tf.paragraphs[0]; p.alignment = align
    if line_spacing:
        p.line_spacing = line_spacing
    _style_run(p.add_run(), text, size, color, bold, font, italic, cs)
    return box


def set_shape_text(shape, text, size, color, bold=False, font=BODY_FONT,
                   align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, italic=False, cs=0):
    tf = shape.text_frame; _frame(tf, valign)
    p = tf.paragraphs[0]; p.alignment = align
    _style_run(p.add_run(), text, size, color, bold, font, italic, cs)


# ============================================================
# SLIDE 1 -- The exercise (dark / graphite)
# ============================================================
s = add_slide(GRAPH)
add_rect(s, 0.7, 0.72, 0.55, 0.09, GOLD)
add_text(s, 0.7, 0.9, 12.0, 1.0, "Reading the calls behind the complaints",
         size=38, bold=True, color=WHITE, font=TITLE_FONT)
add_text(s, 0.7, 1.95, 11.9, 0.7,
         "We know WHERE complaints run high. This exercise reads the actual calls to learn "
         "WHY — and separates what we can fix from what we can’t.",
         size=17, color=GCCC, line_spacing=1.05)

steps = [
    ("1", "Flag the gap",      "where complaints & CRC run high, by call type × site"),
    ("2", "Sample the calls",  "complaint and non-complaint calls, balanced"),
    ("3", "Read blind",        "an LLM scores each transcript with the outcome hidden"),
    ("4", "Surface the drivers","plain-language causes + verbatim customer evidence"),
]
x, cw, gap = 0.7, 2.84, 0.27
for n, t, d in steps:
    add_rect(s, x, 3.25, cw, 2.40, CARD, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_rect(s, x, 3.25, cw, 0.07, GOLD)                         # gold cap
    add_text(s, x + 0.18, 3.45, cw - 0.36, 0.62, n, size=30, bold=True,
             color=GOLD, font=TITLE_FONT)
    add_text(s, x + 0.18, 4.12, cw - 0.36, 0.50, t, size=16, bold=True, color=WHITE)
    add_text(s, x + 0.18, 4.62, cw - 0.36, 0.95, d, size=12, color=GCCC, line_spacing=1.03)
    x += cw + gap

add_text(s, 0.7, 6.55, 12.0, 0.5,
         "A customer-experience lens — grounded in the actual conversations, not just the rates.",
         size=16, italic=True, color=GOLD)
s.notes_slide.notes_text_frame.text = (
    "Framing: this exercise finds and explains BAD CUSTOMER EXPERIENCES, not reviews agents. "
    "We start from WHERE complaints are high (rates by call type x site) and read the actual "
    "calls, blind to the outcome, to learn WHY -- then sort fixable drivers from structural ones.")


# ============================================================
# SLIDE 2 -- What the calls reveal (light)
# ============================================================
s = add_slide(WHITE)
add_rect(s, 0.7, 0.42, 0.55, 0.09, GOLD)
add_text(s, 0.7, 0.58, 12.0, 0.8, "What the calls reveal",
         size=34, bold=True, color=GRAPH, font=TITLE_FONT)
add_text(s, 0.7, 1.42, 12.0, 0.45,
         "CWC sets the pace in two call types — and reading the calls shows the gap is in "
         "how they are handled.",
         size=15, italic=True, color=MUTE)


def rate_panel(y, ctype, good_site, good_pct, bad_site, bad_pct, delta):
    add_rect(s, 0.7, y, 5.95, 1.66, OFF, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, 0.95, y + 0.13, 3.9, 0.33, ctype + "  —  complaint rate",
             size=13.5, bold=True, color=GRAPH)
    chip = add_rect(s, 5.55, y + 0.13, 0.95, 0.38, GOLD,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    set_shape_text(chip, delta, 12.5, GRAPH, bold=True)
    # good / lower-complaint
    add_text(s, 0.95, y + 0.52, 2.3, 0.66, good_pct, size=32, bold=True,
             color=GRAPH, font=TITLE_FONT)
    add_text(s, 0.97, y + 1.19, 2.6, 0.3, good_site + " · lower-complaint",
             size=10.5, color=MUTE)
    # bad / higher-complaint  (gold underline marks the worse side)
    add_text(s, 3.55, y + 0.52, 2.3, 0.66, bad_pct, size=32, bold=True,
             color=GRAPH, font=TITLE_FONT)
    add_rect(s, 3.57, y + 1.14, 1.15, 0.06, GOLD)
    add_text(s, 3.57, y + 1.19, 2.9, 0.3, bad_site + " · higher-complaint",
             size=10.5, color=MUTE)


rate_panel(1.95, "CLIENT SERVICES", "CWC", "30%", "HYDERABAD", "61%", "+31 pts")
rate_panel(3.75, "FRAUD",           "CWC", "59%", "PHILIPPINES", "72%", "+13 pts")
add_text(s, 0.7, 5.5, 5.95, 0.3,
         "Good vs. high-complaint sites compared like-for-like, on the same call type.",
         size=10.5, italic=True, color=MUTE)

# drivers (right)
add_text(s, 7.05, 1.95, 5.6, 0.4, "What drives the worse experience",
         size=16, bold=True, color=GRAPH)
drivers = [
    ("Issues left unresolved — or handed off",
     "“there’s nothing we can do — call collections”"),
    ("Weaker ownership of the fix",
     "fewer “I’ll take care of it” moments on the call"),
    ("Customers escalate more often",
     "e.g. Fraud escalation 0.69 vs 0.46 at CWC"),
]
dy = 2.5
for h, d in drivers:
    add_rect(s, 7.05, dy + 0.04, 0.2, 0.2, GOLD)
    add_text(s, 7.4, dy - 0.06, 5.3, 0.4, h, size=14.5, bold=True, color=GRAPH)
    add_text(s, 7.4, dy + 0.32, 5.3, 0.38, d, size=11.5, color=MUTE, italic=True)
    dy += 0.97

# what the customer heard (bottom)
add_text(s, 0.7, 5.92, 12.0, 0.36, "What the customer heard", size=15, bold=True, color=GRAPH)
# WORSE -- gold left border
add_rect(s, 0.7, 6.3, 5.95, 0.95, OFF, line=LINE, line_w=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
add_rect(s, 0.7, 6.3, 0.1, 0.95, GOLD)
add_text(s, 0.95, 6.4, 3.0, 0.26, "WORSE", size=10, bold=True, color=GRAPH, cs=1)
add_text(s, 0.95, 6.66, 5.6, 0.5,
         "“There’s nothing we can do — you need to speak to collections.”",
         size=13, italic=True, color=GRAPH)
# BETTER -- graphite left border
add_rect(s, 6.85, 6.3, 5.8, 0.95, OFF, line=LINE, line_w=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
add_rect(s, 6.85, 6.3, 0.1, 0.95, GRAPH)
add_text(s, 7.1, 6.4, 3.0, 0.26, "BETTER", size=10, bold=True, color=GRAPH, cs=1)
add_text(s, 7.1, 6.66, 5.45, 0.5,
         "“I’m removing that entire amount — you’re all set.”",
         size=13, italic=True, color=GRAPH)
s.notes_slide.notes_text_frame.text = (
    "CLIENT SERVICES: CWC 30% vs HYDERABAD 61% complaint (CRC 10% vs 32%). "
    "FRAUD: CWC 59% vs PHILIPPINES 72%. Read like-for-like (same call type, complaint calls), "
    "the high-complaint sites score lower on clarity, ownership and resolution, and show more "
    "escalation. Drivers are recurring and fixable. Quotes are de-identified exemplars from the "
    "blind read.")


# ============================================================
# SLIDE 3 -- Fixable vs structural (light)
# ============================================================
s = add_slide(WHITE)
add_rect(s, 0.7, 0.42, 0.55, 0.09, GOLD)
add_text(s, 0.7, 0.58, 12.0, 0.8, "Telling a fixable gap from a structural one",
         size=34, bold=True, color=GRAPH, font=TITLE_FONT)
add_text(s, 0.7, 1.42, 12.0, 0.4, "The same method flags gaps we should NOT coach to.",
         size=15, italic=True, color=MUTE)

# left: counter-case as a 3-step logic
add_rect(s, 0.7, 1.95, 6.05, 4.35, OFF, line=LINE, line_w=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
add_text(s, 0.98, 2.12, 5.5, 0.36, "CUSTOMER SERVICE — a counter-case",
         size=14, bold=True, color=GRAPH)
logic = [
    ("1", "The rates",
     "CANTON 42% vs RAPID CITY 59% complaint — on rates alone, RAPID CITY looks worse.", False),
    ("2", "The calls",
     "Read blind, RAPID CITY handled calls as well or better (overall 3.04 vs 2.94). "
     "CANTON is only a handful of agents.", False),
    ("3", "The verdict",
     "This gap is structural — call & queue mix — not how the call was handled. "
     "No coaching lever here.", True),
]
ly = 2.62
for n, t, d, em in logic:
    circ = add_rect(s, 0.98, ly, 0.34, 0.34, GOLD, shape=MSO_SHAPE.OVAL)
    set_shape_text(circ, n, 15, GRAPH, bold=True, font=TITLE_FONT)
    add_text(s, 1.48, ly - 0.02, 5.0, 0.32, t, size=14, bold=True, color=GRAPH)
    add_text(s, 1.48, ly + 0.3, 5.1, 0.85, d, size=12.5,
             color=(GRAPH if em else MUTE), italic=em, bold=em, line_spacing=1.03)
    ly += 1.18

# right: why it matters + what's next
add_text(s, 7.15, 1.95, 5.5, 0.36, "Why it matters", size=16, bold=True, color=GRAPH)
add_text(s, 7.15, 2.38, 5.5, 1.0,
         "The method separates what coaching & process can fix from what’s structural "
         "— so effort lands where it actually moves the needle.",
         size=13.5, color=MUTE, line_spacing=1.05)
add_text(s, 7.15, 3.5, 5.5, 0.36, "What’s next", size=16, bold=True, color=GRAPH)
nexts = [
    "Scale monthly across call types & sites",
    "Route the fixable drivers — resolve on the call, clear next steps, "
    "early de-escalation — into QA & process",
    "Treat as a customer-experience input — not a scorecard",
]
ny = 3.98
for t in nexts:
    add_rect(s, 7.15, ny + 0.05, 0.18, 0.18, GOLD)
    add_text(s, 7.48, ny - 0.05, 5.15, 0.6, t, size=13.5, color=GRAPH, line_spacing=1.03)
    ny += 0.82 if len(t) > 60 else 0.5

add_text(s, 0.7, 6.62, 12.0, 0.4,
         "Find the bad experiences, explain them, fix the fixable.",
         size=16, italic=True, bold=True, color=GRAPH)
s.notes_slide.notes_text_frame.text = (
    "The CANTON/RAPID CITY case is the integrity point: a LOWER-complaint queue whose calls were "
    "NOT handled better (RAPID CITY, the higher-complaint queue, scored the same or better). "
    "CANTON is a very small team, so its low rate reflects mix, not a repeatable handling "
    "advantage. Including this shows the exercise distinguishes fixable experience drivers from "
    "structural ones -- it is a customer-experience input, not an agent scorecard.")


OUT = "ECCE_Customer_Experience_DeepDive.pptx"
prs.save(OUT)
print("Wrote:", OUT)
