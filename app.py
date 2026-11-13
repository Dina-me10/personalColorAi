# VEB-SAHIFA: Gradio yordamida rasm yuklab, faslni aniqlash (multfilmga o'xshash dizayn).
# Ishga tushirish:  python app.py
# Keyin brauzerda oching:  http://127.0.0.1:7860

import cv2
import gradio as gr

# main.py dagi tayyor funksiya va ma'lumotlarni olamiz.
# main.py ichidagi "if __name__ == '__main__':" bloki bu yerda bajarilmaydi.
from main import analyze_image, SEASON_COLORS_BGR


# ---- Fasllar haqida ma'lumot (faqat ko'rinish uchun) ----

SEASON_EMOJI = {
    "Spring Warm": "🌷",
    "Summer Cool": "🌸",
    "Autumn Warm": "🍂",
    "Winter Cool": "❄️",
}

# Har bir fasl nishonchasi (badge) uchun o'z pastel rangi.
SEASON_BADGE_COLOR = {
    "Spring Warm": "#FFE7A8",
    "Summer Cool": "#FFD3E6",
    "Autumn Warm": "#F9C9A2",
    "Winter Cool": "#CFE6FF",
}

SEASON_SENTENCE = {
    "Spring Warm": "You're bright and sunny like spring — cheerful, warm colors make your face glow!",
    "Summer Cool": "You're soft and breezy like a summer morning — gentle, cool pastels suit you beautifully!",
    "Autumn Warm": "You're rich and cozy like autumn — earthy, golden tones make you shine!",
    "Winter Cool": "You're crisp and striking like winter — bold, clear, high-contrast colors look amazing on you!",
}

# Ranglarning inglizcha nomlari — main.py dagi SEASON_COLORS_BGR bilan bir xil tartibda.
# (main.py dagi SEASON_TIPS o'zbekcha qoladi, chunki u terminalda chiqadi.)
SEASON_COLOR_NAMES = {
    "Spring Warm": ["coral", "peach", "warm yellow", "light green", "gold"],
    "Summer Cool": ["lavender", "light pink", "sky blue", "blue-gray", "silver"],
    "Autumn Warm": ["terracotta", "mustard", "olive green", "brown", "gold"],
    "Winter Cool": ["black", "white", "navy", "fuchsia", "silver"],
}


# OpenCV rangi (B, G, R) -> veb rangi "#RRGGBB".
def bgr_to_hex(bgr):
    blue, green, red = bgr
    return "#%02X%02X%02X" % (red, green, blue)


# ---- HTML bo'laklari ----

# Maskot: o'zimiz chizgan kulib turgan bo'yoq palitrasi (SVG — kod bilan chizilgan rasm).
MASCOT_SVG = """
<svg class="mascot" viewBox="0 0 120 120" width="104" height="104" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M62 12 C97 10 116 36 112 63 C108 90 86 106 60 105 C45 104 40 96 44 87
           C47 79 41 73 31 77 C18 82 7 70 9 55 C12 29 32 13 62 12 Z"
        fill="#FFF3DC" stroke="#2b2b2b" stroke-width="4" stroke-linejoin="round"/>
  <circle cx="30" cy="56" r="7" fill="#FFE4D6" stroke="#2b2b2b" stroke-width="3"/>
  <circle cx="45" cy="30" r="8" fill="#FF7A6B" stroke="#2b2b2b" stroke-width="3"/>
  <circle cx="69" cy="23" r="8" fill="#FFD25E" stroke="#2b2b2b" stroke-width="3"/>
  <circle cx="92" cy="32" r="8" fill="#8FD9B6" stroke="#2b2b2b" stroke-width="3"/>
  <circle cx="104" cy="55" r="7" fill="#A9B8FF" stroke="#2b2b2b" stroke-width="3"/>
  <circle cx="64" cy="62" r="4.5" fill="#2b2b2b"/>
  <circle cx="84" cy="62" r="4.5" fill="#2b2b2b"/>
  <circle cx="65.5" cy="60.5" r="1.5" fill="#fff"/>
  <circle cx="85.5" cy="60.5" r="1.5" fill="#fff"/>
  <path d="M67 73 Q74 81 81 73" fill="none" stroke="#2b2b2b" stroke-width="3.5" stroke-linecap="round"/>
  <ellipse cx="56" cy="72" rx="6" ry="3.5" fill="#FF9EAA" opacity="0.85"/>
  <ellipse cx="92" cy="72" rx="6" ry="3.5" fill="#FF9EAA" opacity="0.85"/>
  <path d="M110 12 l2.5 6 6 2.5 -6 2.5 -2.5 6 -2.5 -6 -6 -2.5 6 -2.5 z" fill="#FFD25E" stroke="#2b2b2b" stroke-width="2" stroke-linejoin="round"/>
</svg>
"""

HEADER_HTML = f"""
<div class="hero">
  {MASCOT_SVG}
  <div>
    <h1 class="title">Personal Color AI</h1>
    <p class="subtitle">Upload a selfie — we'll find your season from your cheek color ✨</p>
  </div>
</div>
"""


# 4 ta fasl nishonchasi. active_season berilsa, o'sha nishoncha ajratib ko'rsatiladi.
def badges_html(active_season=None):
    html = '<div class="badges">'
    for season, emoji in SEASON_EMOJI.items():
        css_class = "badge"
        if season == active_season:
            css_class += " active"
        elif active_season is not None:
            css_class += " dim"
        color = SEASON_BADGE_COLOR[season]
        html += f'<span class="{css_class}" style="background:{color}">{emoji} {season}</span>'
    html += "</div>"
    return html


PLACEHOLDER_HTML = """
<div class="card placeholder">
  <div class="big-emoji">🎨</div>
  <p>Your result will appear here 🎨</p>
</div>
"""

NO_FACE_HTML = """
<div class="card placeholder">
  <div class="big-emoji">🙈</div>
  <p>No face found. Try a well-lit photo where you're looking straight at the camera.</p>
</div>
"""


# Natija kartasi: fasl nomi, jumla va 5 ta mos rang (faqat ko'rsatish uchun).
def result_card_html(info):
    season = info["season"]
    names = SEASON_COLOR_NAMES[season]
    colors = SEASON_COLORS_BGR[season]

    swatches = ""
    for name, bgr in zip(names, colors):
        hex_color = bgr_to_hex(bgr)
        swatches += (f'<div class="swatch"><span class="dot" style="background:{hex_color}"></span>'
                     f'<span class="swatch-name">{name}</span></div>')

    note = ""
    if info["uncertain"]:
        note = '<p class="note">⚠️ Your values are very close to the cutoff — try another photo in different lighting.</p>'

    color = SEASON_BADGE_COLOR[season]
    return f"""
<div class="card">
  <div class="season-name" style="background:{color}">{SEASON_EMOJI[season]} {season}</div>
  <p class="sentence">{SEASON_SENTENCE[season]}</p>
  <div class="swatches">{swatches}</div>
  <p class="numbers">L* = {round(info["L"], 1)} · Hue = {round(info["hue"], 1)}°</p>
  {note}
</div>
"""


# ---- Asosiy funksiya ----

# Rasm yuklanganda yoki "Analyze" bosilganda ishlaydi.
# Qaytaradi: natija rasmi, karta, nishonchalar.
def predict(image_rgb):
    # Hali rasm yuklanmagan bo'lsa.
    if image_rgb is None:
        return None, PLACEHOLDER_HTML, badges_html()

    # Gradio rasmni RGB tartibida beradi, OpenCV esa BGR kutadi (1-qadamni eslang!).
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    result = analyze_image(image_bgr)
    if result is None:
        return None, NO_FACE_HTML, badges_html()

    # Natija rasmi — yuklangan rasmning o'zi: doirachalarsiz, bo'yalmagan, toza.
    return image_rgb, result_card_html(result), badges_html(result["season"])


# ---- Dizayn (CSS) ----

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');

:root {
  --ink: #2b2b2b;
  --coral: #FF7A6B;
  --cream: #FFF8EC;
}

/* Fon: krem -> shaftoli gradient */
body, gradio-app, .gradio-container {
  background: linear-gradient(160deg, #FFF8EC 0%, #FFE9DA 50%, #FFDCCF 100%) !important;
  background-attachment: fixed !important;
}
.gradio-container {
  max-width: 1080px !important;
  margin: 0 auto !important;
  padding-top: 24px !important;
}
.gradio-container, .gradio-container * {
  font-family: 'Fredoka', 'Baloo 2', ui-rounded, system-ui, sans-serif !important;
  color: var(--ink);
}

/* Gradio pastki yozuvini (footer) yashiramiz */
footer { display: none !important; }

/* Sarlavha va maskot */
.hero {
  display: flex; align-items: center; justify-content: center; gap: 18px;
  flex-wrap: wrap; text-align: left; margin-bottom: 4px;
}
.mascot { flex: none; animation: wiggle 3.5s ease-in-out infinite; }
@keyframes wiggle {
  0%, 100% { transform: rotate(-4deg); }
  50% { transform: rotate(4deg) translateY(-3px); }
}
.title {
  font-size: 52px !important; font-weight: 700 !important; margin: 0 !important;
  line-height: 1.05; letter-spacing: -0.5px;
  text-shadow: 3px 3px 0 #FFC2B4;
}
.subtitle { font-size: 19px; margin: 6px 0 0 !important; opacity: 0.85; }

/* Fasl nishonchalari */
.badges { display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; margin: 10px 0 6px; }
.badge {
  border: 3px solid var(--ink); border-radius: 999px; padding: 6px 16px;
  font-weight: 600; font-size: 16px; box-shadow: 3px 3px 0 var(--ink);
  transition: transform .2s, opacity .2s, box-shadow .2s;
}
.badge.active { transform: scale(1.12) rotate(-2deg); box-shadow: 5px 5px 0 var(--ink); }
.badge.dim { opacity: 0.45; }

/* "Stiker" kartalar */
.sticker {
  background: #fff !important;
  border: 3px solid var(--ink) !important;
  border-radius: 28px !important;
  box-shadow: 6px 6px 0 var(--ink) !important;
  padding: 18px !important;
  gap: 16px !important;
}
.sticker > .block, .sticker .image-frame, .sticker .block.html-container {
  border-radius: 20px !important;
}
.sticker > .block:not(.html-container) {
  border: 3px dashed #F3B9A9 !important;
  background: var(--cream) !important;
  box-shadow: none !important;
  overflow: hidden;
}
.sticker .html-container { border: none !important; background: transparent !important; padding: 0 !important; }
.sticker label span, .sticker [data-testid="block-label"] {
  background: #FFE9DA !important; border-radius: 999px !important; font-weight: 600 !important;
}

/* "Analyze" tugmasi */
#go-btn {
  background: var(--coral) !important; color: #fff !important;
  border: 3px solid var(--ink) !important; border-radius: 999px !important;
  box-shadow: 5px 5px 0 var(--ink) !important;
  font-size: 24px !important; font-weight: 700 !important; padding: 14px 0 !important;
  transition: box-shadow .1s, transform .1s;
}
#go-btn:hover { animation: bounce .55s ease; }
#go-btn:active { transform: translate(4px, 4px); box-shadow: 1px 1px 0 var(--ink) !important; }
@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  35% { transform: translateY(-7px) scale(1.03); }
  65% { transform: translateY(2px); }
}

/* Natija kartasi */
.card { text-align: center; padding: 6px 4px; }
.season-name {
  display: inline-block; font-size: 34px; font-weight: 700;
  border: 3px solid var(--ink); border-radius: 999px; padding: 6px 26px;
  box-shadow: 4px 4px 0 var(--ink); transform: rotate(-1.5deg); margin: 4px 0 12px;
}
.sentence { font-size: 18px; margin: 4px 8px 16px !important; line-height: 1.4; }
.swatches { display: flex; justify-content: center; flex-wrap: wrap; gap: 14px; }
.swatch { display: flex; flex-direction: column; align-items: center; width: 74px; }
.dot {
  display: block; width: 58px; height: 58px; border-radius: 50%; padding: 0;
  border: 3px solid var(--ink); box-shadow: 3px 3px 0 var(--ink);
}
.swatch-name { font-size: 13px; margin-top: 8px; line-height: 1.2; }
.numbers { font-size: 14px; opacity: 0.65; margin: 16px 0 0 !important; }
.note { font-size: 14px; background: #FFF3C4; border-radius: 12px; padding: 6px 10px; margin-top: 10px !important; }
.placeholder { padding: 36px 10px; }
.placeholder p { font-size: 20px; font-weight: 500; }
.big-emoji { font-size: 56px; animation: wiggle 3s ease-in-out infinite; display: inline-block; }
"""

# Sahifa doim och (light) rejimda ochilsin — qorong'i rejimda pastel dizayn buziladi.
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.href = url.href;
  }
}
"""


# ---- Sahifa tuzilishi ----

with gr.Blocks(title="Personal Color AI", css=CSS, js=FORCE_LIGHT_JS) as demo:
    gr.HTML(HEADER_HTML)
    badges = gr.HTML(badges_html())

    with gr.Row(equal_height=False):
        # Chap karta: rasm yuklash
        with gr.Column(elem_classes="sticker"):
            input_image = gr.Image(label="Your photo 📸", type="numpy", sources=["upload", "webcam"])
            button = gr.Button("Analyze ✨", elem_id="go-btn")
        # O'ng karta: natija
        with gr.Column(elem_classes="sticker"):
            output_image = gr.Image(label="Result 🖼️", interactive=False)
            result_card = gr.HTML(PLACEHOLDER_HTML)

    outputs = [output_image, result_card, badges]
    # Tugma bosilganda yoki rasm yuklanganda predict() ishlaydi.
    button.click(predict, inputs=input_image, outputs=outputs)
    input_image.upload(predict, inputs=input_image, outputs=outputs)
    # Rasm o'chirilsa, natijani ham tozalaymiz.
    input_image.clear(lambda: predict(None), outputs=outputs)


if __name__ == "__main__":
    demo.launch()
