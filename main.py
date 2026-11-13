# ASOSIY DASTUR: barcha qadamlarni (1-6) bitta faylda bajaradi.
# Ishlatish:
#   python main.py                 -> selfie.jpg ni tahlil qiladi
#   python main.py rasm.jpg        -> istalgan rasmni tahlil qiladi
# Natija rasmi "<rasm nomi>_result.jpg" sifatida saqlanadi.

import os
import sys

import cv2
import numpy as np


# Model fayli main.py bilan bir papkada. __file__ — shu faylning yo'li.
# Shunday qilsak, dasturni boshqa papkadan ishga tushirsangiz ham model topiladi.
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx")


# Yuzni topish (3-qadamdagi bilan bir xil).
def detect_face(image):
    height, width, channels = image.shape
    detector = cv2.FaceDetectorYN.create(model_path, "", (width, height))
    detector.setScoreThreshold(0.9)
    _, faces = detector.detect(image)
    if faces is None:
        return None
    return faces[0]


# Yonoqlarni topish (3-qadamdagi bilan bir xil).
def find_cheeks(face):
    right_eye_x = face[4]
    right_eye_y = face[5]
    left_eye_x = face[6]
    left_eye_y = face[7]
    mouth_right_y = face[11]
    mouth_left_y = face[13]

    right_cheek_x = right_eye_x
    right_cheek_y = (right_eye_y + mouth_right_y) / 2
    left_cheek_x = left_eye_x
    left_cheek_y = (left_eye_y + mouth_left_y) / 2

    eye_distance = left_eye_x - right_eye_x
    radius = eye_distance * 0.15

    right_cheek = (int(right_cheek_x), int(right_cheek_y))
    left_cheek = (int(left_cheek_x), int(left_cheek_y))
    return right_cheek, left_cheek, int(radius)


# YANGI: ikkala yonoq doirasi ichidagi piksellarni olib, LAB qiymatlarini qaytaradi.
def get_cheek_lab(image, right_cheek, left_cheek, radius):
    height, width, channels = image.shape

    # "Niqob" (mask) — rasm bilan bir xil o'lchamdagi qora (0) rasm.
    # Unga yonoqlar o'rnida oq (255) to'ldirilgan doiralar chizamiz.
    # Keyin faqat oq joylardagi piksellarni olamiz.
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, right_cheek, radius, 255, -1)
    cv2.circle(mask, left_cheek, radius, 255, -1)

    # Rasmni 0..255 butun sonlardan 0..1 kasr sonlarga o'tkazamiz.
    # Sababi: kasr rasmda OpenCV haqiqiy LAB qiymatlarini beradi (L* 0..100).
    # Butun son (uint8) rasmda esa qiymatlar boshqa shkalaga siqiladi va chalkashlik bo'ladi.
    image_float = image.astype(np.float32) / 255.0

    # BGR -> LAB. OpenCV'ga rasm BGR tartibida ekanini aytamiz (1-qadamni eslang!).
    lab = cv2.cvtColor(image_float, cv2.COLOR_BGR2LAB)

    # Faqat niqob oq bo'lgan joylardagi piksellarni olamiz.
    # Natija — N ta qator, har birida 3 ta son: [L, a, b].
    cheek_pixels = lab[mask == 255]

    # Median — o'rtadagi qiymat. O'rtacha (mean) qiymatdan farqli o'laroq,
    # bir nechta g'alati piksel (dog', yaltirash, soya) natijani buzmaydi.
    # axis=0 — har bir ustun (L, a, b) uchun alohida hisoblash.
    median_lab = np.median(cheek_pixels, axis=0)

    L = float(median_lab[0])
    a = float(median_lab[1])
    b = float(median_lab[2])
    return L, a, b, len(cheek_pixels)


# Hue (rang tusi) burchagini hisoblaydi — 0 dan 360 gradusgacha.
def hue_angle(a, b):
    # arctan2(b, a) — (a, b) nuqtaning burchagini radianda beradi.
    # degrees() — radianni gradusga aylantiradi.
    angle = np.degrees(np.arctan2(b, a))
    # Burchak manfiy chiqsa, 360 qo'shib musbat qilamiz.
    if angle < 0:
        angle = angle + 360
    return float(angle)




# ---- YANGI: faslni aniqlash ----

# Chegaralar (threshold). Bular qat'iy ilmiy qiymatlar emas — oddiy taxminiy qoidalar.
# O'z rasmlaringizda sinab ko'rib, kerak bo'lsa o'zgartiring.
#
# Hue burchagi teri uchun odatda 40..70 gradus atrofida bo'ladi:
#   kichik (pushtiroq)  -> sovuq (cool) ohang
#   katta  (sarg'ishroq) -> iliq (warm) ohang
HUE_THRESHOLD = 55
#
# L* — yorug'lik. Yuqori bo'lsa teri ochroq (light), past bo'lsa to'qroq (deep).
LIGHTNESS_THRESHOLD = 65.0


# 1) Iliq yoki sovuq?
def get_undertone(hue):
    if hue >= HUE_THRESHOLD:
        return "warm"
    return "cool"


# 2) Och yoki to'q?
def get_depth(L):
    if L >= LIGHTNESS_THRESHOLD:
        return "light"
    return "deep"


# 3) Ikkalasini birlashtirib, 4 fasldan birini tanlaymiz:
#             light          deep
#   warm   Spring Warm    Autumn Warm
#   cool   Summer Cool    Winter Cool
def get_season(undertone, depth):
    if undertone == "warm" and depth == "light":
        return "Spring Warm"
    if undertone == "warm" and depth == "deep":
        return "Autumn Warm"
    if undertone == "cool" and depth == "light":
        return "Summer Cool"
    return "Winter Cool"


# Har bir faslga mos keladigan ranglar — foydalanuvchiga maslahat sifatida.
SEASON_TIPS = {
    "Spring Warm": "marjon, shaftoli, iliq sariq, och yashil, oltin",
    "Summer Cool": "lavanda, och pushti, osmon ko'k, kulrang-ko'k, kumush",
    "Autumn Warm": "terrakota, xantal sariq, zaytun yashil, jigarrang, oltin",
    "Winter Cool": "qora, oq, to'q ko'k, yorqin qizil (fuksiya), kumush",
}


# ---- YANGI: natijani rasmga chizish ----

# Har bir fasl uchun 5 ta rang. OpenCV ranglarni BGR tartibida oladi (ko'k, yashil, qizil)!
SEASON_COLORS_BGR = {
    "Spring Warm": [(80, 127, 255), (185, 218, 255), (90, 215, 255), (144, 238, 144), (55, 175, 212)],
    "Summer Cool": [(250, 230, 230), (203, 192, 255), (235, 206, 135), (153, 136, 119), (192, 192, 192)],
    "Autumn Warm": [(91, 114, 204), (88, 173, 225), (35, 128, 128), (33, 67, 101), (32, 165, 218)],
    "Winter Cool": [(0, 0, 0), (255, 255, 255), (128, 0, 0), (255, 0, 255), (192, 192, 192)],
}


def draw_result(image, right_cheek, left_cheek, radius, season, L, hue, undertone, depth):
    # Asl rasmni buzmaslik uchun nusxasini olamiz.
    result = image.copy()
    height, width, channels = result.shape

    # 1) Yonoqlar atrofida yashil doira (o'lchagan joyimizni ko'rsatish uchun).
    cv2.circle(result, right_cheek, radius, (0, 255, 0), 2)
    cv2.circle(result, left_cheek, radius, (0, 255, 0), 2)

    # 2) Rasm pastiga oq "panel" qo'shamiz — matn va ranglar shu yerda bo'ladi.
    # Panel balandligi rasm kengligiga qarab o'zgaradi, kichik rasmlarda ham sig'sin.
    panel_height = max(140, width // 3)
    panel = np.full((panel_height, width, 3), 255, dtype=np.uint8)

    # Shrift o'lchami ham rasm kengligiga moslashadi.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = width / 600
    thickness = max(1, int(scale * 2))
    margin = int(width * 0.04)

    # 3) Fasl nomi (katta) va o'lchangan qiymatlar (kichik).
    # Eslatma: cv2.putText faqat lotin harflarini chiza oladi.
    title_y = int(panel_height * 0.25)
    cv2.putText(panel, season, (margin, title_y), font, scale * 1.2, (40, 40, 40), thickness + 1)
    info = "L*=" + str(round(L, 1)) + " (" + depth + ")   hue=" + str(round(hue, 1)) + " (" + undertone + ")"
    info_y = int(panel_height * 0.42)
    cv2.putText(panel, info, (margin, info_y), font, scale * 0.55, (90, 90, 90), max(1, thickness - 1))

    # 4) 5 ta rang kvadratchasi — panel kengligi bo'ylab teng taqsimlanadi.
    colors = SEASON_COLORS_BGR[season]
    gap = margin // 2
    swatch_size = (width - 2 * margin - gap * (len(colors) - 1)) // len(colors)
    swatch_size = min(swatch_size, int(panel_height * 0.45))
    top = int(panel_height * 0.5)
    for i, color in enumerate(colors):
        left = margin + i * (swatch_size + gap)
        # -1 — to'ldirilgan to'rtburchak; keyin kulrang chegara (oq rang ko'rinsin).
        cv2.rectangle(panel, (left, top), (left + swatch_size, top + swatch_size), color, -1)
        cv2.rectangle(panel, (left, top), (left + swatch_size, top + swatch_size), (180, 180, 180), 1)

    # 5) Rasm va panelni ustma-ust (vertikal) birlashtiramiz.
    return np.vstack([result, panel])


def analyze(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Xato: rasm topilmadi yoki ochib bo'lmadi:", image_path)
        return False

    face = detect_face(image)
    if face is None:
        print("Yuz topilmadi:", image_path)
        print("Yuzingiz to'g'ridan-to'g'ri va yaxshi yoritilgan rasm bilan sinab ko'ring.")
        return False

    right_cheek, left_cheek, radius = find_cheeks(face)
    L, a, b, pixel_count = get_cheek_lab(image, right_cheek, left_cheek, radius)
    hue = hue_angle(a, b)

    undertone = get_undertone(hue)
    depth = get_depth(L)
    season = get_season(undertone, depth)

    # "rasm.jpg" -> "rasm_result.jpg"
    # os.path.splitext nomni va kengaytmani (.jpg) ajratadi.
    name, extension = os.path.splitext(image_path)
    output_path = name + "_result.jpg"
    result = draw_result(image, right_cheek, left_cheek, radius, season, L, hue, undertone, depth)
    cv2.imwrite(output_path, result)

    print("Rasm:", image_path)
    print("L* (yorug'lik):", round(L, 1), "->", depth)
    print("Hue burchagi:  ", round(hue, 1), "->", undertone)
    print("Sizning faslingiz:", season)
    print("Sizga mos ranglar:", SEASON_TIPS[season])
    if abs(hue - HUE_THRESHOLD) < 3 or abs(L - LIGHTNESS_THRESHOLD) < 3:
        print("Eslatma: qiymatlar chegaraga juda yaqin, natija aniq emas.")
    print("Natija saqlandi:", output_path)
    return True


# ---- Asosiy dastur ----

# sys.argv — terminalda yozilgan so'zlar ro'yxati: ["main.py", "rasm.jpg", ...]
# Agar rasm nomi berilmagan bo'lsa, selfie.jpg ishlatiladi.
# Bir nechta rasm berilsa, har birini navbat bilan tahlil qiladi.
image_paths = sys.argv[1:]
if len(image_paths) == 0:
    image_paths = ["selfie.jpg"]

all_ok = True
for i, path in enumerate(image_paths):
    if i > 0:
        print()
    if not analyze(path):
        all_ok = False

if not all_ok:
    raise SystemExit(1)
