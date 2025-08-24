from PIL import Image, ImageDraw, ImageFont, ImageOps
import os

# ---------------- Configuración ----------------
# Para cada figura: (ruta, etiqueta para la banda izquierda, fila a conservar)
figs = [
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/Standard_FNO_64x64.png",                    "Standard FNO",                 "top"),
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/Smaller_FNO_shared_weights_64x64.png",      "Smaller FNO (shared)",         "bottom"),
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/Ensemble_FNO_64x64.png",                    "Ensemble FNO",                 "top"),
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/Enhanced_Smaller_FNO_Better_training_64x64.png", "Enhanced Smaller FNO",   "bottom"),
    ("/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/figs/Optimized_95_Target_FNO_64x64.png",         "Optimized Target FNO",         "bottom"),
]

# Ancho de la banda de etiquetas (izquierda)
LABEL_BAND = 220

# Reescalar ancho final de cada fila (None = mantener ancho original tras recorte)
TARGET_ROW_WIDTH = 1400

# Tipografía
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 58)
except:
    FONT = ImageFont.load_default()

# ---------------- Utilidades ----------------
def auto_trim(im, bg=(255, 255, 255), tol=8):
    """
    Recorta márgenes casi blancos. Ajusta tol si tus márgenes no son blancos puros.
    """
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg_im = Image.new("RGB", im.size, bg)
    diff = ImageChops.difference(im, bg_im)
    # Amplifica la diferencia para mejorar bounding box
    bbox = diff.convert("L").point(lambda x: 255 if x > tol else 0).getbbox()
    return im.crop(bbox) if bbox else im

def extract_row(im, which="top"):
    """
    Después de recortar márgenes, divide la imagen en dos filas y devuelve
    la fila indicada ('top' o 'bottom'). Si tus figuras no son exactamente
    mitad/mitad, ajusta la 'gap' o usa un ratio.
    """
    w, h = im.size
    # A veces hay una línea/espacio entre filas; dejamos margen de seguridad
    mid = h // 2
    pad = int(0.02 * h)  # 2% de la altura para absorber separadores
    if which == "top":
        box = (0, 0, w, mid + pad)
    else:  # "bottom"
        box = (0, mid - pad, w, h)
    row = im.crop(box)
    # Segundo recorte suave por si quedaron márgenes blancos adicionales
    row = trim_whitespace_soft(row)
    return row

def trim_whitespace_soft(im, thresh=245):
    """
    Trim por umbral de luminosidad (casi blanco). No tan agresivo como auto_trim.
    """
    if im.mode != "L":
        gray = im.convert("L")
    else:
        gray = im
    # Binarizamos: fondo blanco (255) vs contenido
    bw = gray.point(lambda x: 0 if x > thresh else 255, "1")
    bbox = bw.getbbox()
    return im.crop(bbox) if bbox else im

# ---------------- Pipeline ----------------
rows = []
for path, label, which in figs:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe la imagen: {path}")
    im = Image.open(path).convert("RGB")

    # 1) Recorta márgenes grandes (bordes, títulos globales)
    try:
        from PIL import ImageChops
    except ImportError:
        ImageChops = None
    if ImageChops is not None:
        im = auto_trim(im)

    # 2) Extrae una sola fila (top/bottom) para evitar duplicados
    row = extract_row(im, which=which)

    # 3) Reescala a un ancho objetivo común (opcional)
    if TARGET_ROW_WIDTH is not None:
        w, h = row.size
        new_h = int(h * (TARGET_ROW_WIDTH / w))
        row = row.resize((TARGET_ROW_WIDTH, new_h), Image.LANCZOS)

    rows.append((row, label))

# 4) Construye figura final vertical con banda de etiquetas
max_w = max(r.size[0] for r, _ in rows)
total_h = sum(r.size[1] for r, _ in rows)
canvas = Image.new("RGB", (LABEL_BAND + max_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

y = 0
for row, label in rows:
    w, h = row.size

    # Banda de texto
    try:
        tw, th = draw.textbbox((0, 0), label, font=FONT)[2:]
    except:
        tw, th = draw.textsize(label, font=FONT)
    tx = (LABEL_BAND - tw) // 2
    ty = y + (h - th) // 2
    draw.text((tx, ty), label, fill=(0, 0, 0), font=FONT)
    draw.line([(LABEL_BAND - 1, y), (LABEL_BAND - 1, y + h)], fill=(220, 220, 220), width=2)

    # Pega la fila
    canvas.paste(row, (LABEL_BAND, y))
    y += h

out_path = "FNO_rows_compact.png"
canvas.save(out_path, dpi=(300, 300))
print(f"Guardado: {out_path}")
