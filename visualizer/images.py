                      

import os
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops

figs = [
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/DeepONet_Model1_random_64x64.png",     "DeepONet Model1 (random)",    "top"),
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/DeepONet_Model1_chebyshev_64x64.png", "DeepONet Model1 (chebyshev)", "top"),
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/DeepONet_Model1_adaptive_64x64.png",  "DeepONet Model1 (adaptive)",  "top"),
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/DeepONet_Model2_uniform_64x64.png",   "DeepONet Model2 (uniform)",   "top"),
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/DON_MSFF_SWA_576_random_64x64.png",   "DON MSFF SWA (random)",       "top"),
]

TARGET_ROW_WIDTH = 1400
FONT_SIZE = 35
FONT_PATHS = ["DejaVuSans.ttf", "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
TEXT_COLOR = (0, 0, 0)
STROKE_COLOR = (255, 255, 255)
STROKE_WIDTH = 0
LABEL_PAD_X = 32
LABEL_PAD_Y = 16
DIVIDER_COLOR = (220, 220, 220)
CANVAS_BG = (255, 255, 255)
OUTPUT_PATH = "deep_rows_compact.png"
OUTPUT_DPI = (300, 300)

def load_font():
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, FONT_SIZE)
        except Exception:
            continue
    return ImageFont.load_default()

FONT = load_font()

def trim_whitespace_soft(im, thresh=245):
    gray = im.convert("L")
    bw = gray.point(lambda x: 0 if x > thresh else 255, "1")
    bbox = bw.getbbox()
    return im.crop(bbox) if bbox else im

def auto_trim(im, bg=(255, 255, 255), tol=8):
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg_im = Image.new("RGB", im.size, bg)
    diff = ImageChops.difference(im, bg_im)
    bbox = diff.convert("L").point(lambda x: 255 if x > tol else 0).getbbox()
    return im.crop(bbox) if bbox else im

def extract_row(im, which="top"):
    w, h = im.size
    mid = h // 2
    pad = int(0.02 * h)
    if which == "top":
        box = (0, 0, w, mid + pad)
    else:
        box = (0, mid - pad, w, h)
    row = im.crop(box)
    row = trim_whitespace_soft(row)
    return row

def measure_text(label, font, stroke_width=0):
    scratch = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(scratch)
    try:
        left, top, right, bottom = d.textbbox((0, 0), label, font=font, stroke_width=stroke_width)
        return right - left, bottom - top
    except Exception:
        w, h = d.textsize(label, font=font)
        return w, h

rows = []
for path, label, which in figs:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    im = Image.open(path).convert("RGB")
    im = auto_trim(im)
    row = extract_row(im, which=which)
    if TARGET_ROW_WIDTH is not None:
        w, h = row.size
        new_h = int(h * (TARGET_ROW_WIDTH / w))
        row = row.resize((TARGET_ROW_WIDTH, new_h), Image.LANCZOS)
    rows.append((row, label))

label_widths = [measure_text(lbl, FONT, STROKE_WIDTH)[0] for _, lbl in rows]
LABEL_BAND = max(label_widths) + 2 * LABEL_PAD_X

max_w = max(r.size[0] for r, _ in rows)
total_h = sum(r.size[1] for r, _ in rows)
canvas = Image.new("RGB", (LABEL_BAND + max_w, total_h), CANVAS_BG)
draw = ImageDraw.Draw(canvas)

y = 0
for row, label in rows:
    w, h = row.size
    tw, th = measure_text(label, FONT, STROKE_WIDTH)
    tx = LABEL_PAD_X + (LABEL_BAND - 2 * LABEL_PAD_X - tw) // 2
    ty = y + (h - th) // 2 + LABEL_PAD_Y
    if STROKE_WIDTH > 0:
        draw.text((tx, ty), label, fill=TEXT_COLOR, font=FONT,
                  stroke_width=STROKE_WIDTH, stroke_fill=STROKE_COLOR)
    else:
        draw.text((tx, ty), label, fill=TEXT_COLOR, font=FONT)
    draw.line([(LABEL_BAND - 1, y), (LABEL_BAND - 1, y + h)], fill=DIVIDER_COLOR, width=2)
    canvas.paste(row, (LABEL_BAND, y))
    y += h

canvas.save(OUTPUT_PATH, dpi=OUTPUT_DPI)
print(f"Saved: {OUTPUT_PATH} (DPI: {OUTPUT_DPI[0]}x{OUTPUT_DPI[1]})")
