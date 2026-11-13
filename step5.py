# 5-QADAM: yonoq rangiga qarab shaxsiy rang faslini (season) aniqlash

import cv2
import numpy as np

image_path = "selfie.jpg"
model_path = "face_detection_yunet_2023mar.onnx"


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


# ---- Asosiy dastur ----

image = cv2.imread(image_path)
if image is None:
    print("Xato: rasm topilmadi:", image_path)
    raise SystemExit(1)

face = detect_face(image)
if face is None:
    print("Yuz topilmadi.")
    raise SystemExit(1)

right_cheek, left_cheek, radius = find_cheeks(face)
L, a, b, pixel_count = get_cheek_lab(image, right_cheek, left_cheek, radius)
hue = hue_angle(a, b)

undertone = get_undertone(hue)
depth = get_depth(L)
season = get_season(undertone, depth)

print("L* (yorug'lik):", round(L, 1), " chegara:", LIGHTNESS_THRESHOLD, "->", depth)
print("Hue burchagi:  ", round(hue, 1), " chegara:", HUE_THRESHOLD, "->", undertone)
print()
print("Sizning faslingiz:", season)
print("Sizga mos ranglar:", SEASON_TIPS[season])

# Agar qiymat chegaraga juda yaqin bo'lsa, natija ishonchsiz — ogohlantiramiz.
if abs(hue - HUE_THRESHOLD) < 3 or abs(L - LIGHTNESS_THRESHOLD) < 3:
    print()
    print("Eslatma: qiymatlar chegaraga juda yaqin, natija aniq emas.")
    print("Boshqa yorug'likda (kunduzgi, deraza yonida) yana bir rasm bilan sinab ko'ring.")
