# 3-QADAM: yuz nuqtalari (landmark) yordamida ikkala yonoqni topish

import cv2

image_path = "selfie.jpg"
model_path = "face_detection_yunet_2023mar.onnx"
output_path = "step3_cheeks.jpg"


# Bu funksiya rasmni oladi va birinchi topilgan yuzni qaytaradi (15 ta son).
# Yuz topilmasa — None qaytaradi. Kod 2-qadamdagi bilan bir xil, faqat funksiyaga o'raldi.
def detect_face(image):
    height, width, channels = image.shape
    detector = cv2.FaceDetectorYN.create(model_path, "", (width, height))
    detector.setScoreThreshold(0.9)
    _, faces = detector.detect(image)
    if faces is None:
        return None
    return faces[0]


# Bu funksiya yuz nuqtalaridan yonoqlarning markazi va doira radiusini hisoblaydi.
def find_cheeks(face):
    # YuNet beradigan 5 ta nuqta (har biri x, y):
    #   4,5   — o'ng ko'z (odamning o'ng ko'zi, rasmda CHAP tomonda)
    #   6,7   — chap ko'z (rasmda O'NG tomonda)
    #   8,9   — burun uchi
    #   10,11 — og'izning o'ng burchagi
    #   12,13 — og'izning chap burchagi
    right_eye_x = face[4]
    right_eye_y = face[5]
    left_eye_x = face[6]
    left_eye_y = face[7]
    mouth_right_y = face[11]
    mouth_left_y = face[13]

    # Yonoq ko'z qorachig'ining to'g'ri ostida joylashgan,
    # balandligi esa ko'z bilan og'iz burchagining o'rtasida (taxminan burun uchi darajasida).
    # Ko'z ostiga juda yaqin olmaymiz — u yerda soya bo'ladi, bu rangni buzadi.
    right_cheek_x = right_eye_x
    right_cheek_y = (right_eye_y + mouth_right_y) / 2

    left_cheek_x = left_eye_x
    left_cheek_y = (left_eye_y + mouth_left_y) / 2

    # Doira o'lchamini ko'zlar orasidagi masofaga bog'laymiz.
    # Shunda yuz rasmda katta yoki kichik bo'lsa ham, doira yuzga mos o'lchamda bo'ladi.
    eye_distance = left_eye_x - right_eye_x
    radius = eye_distance * 0.15

    # Piksel koordinatalari butun son bo'lishi kerak.
    right_cheek = (int(right_cheek_x), int(right_cheek_y))
    left_cheek = (int(left_cheek_x), int(left_cheek_y))
    return right_cheek, left_cheek, int(radius)


# ---- Asosiy dastur ----

image = cv2.imread(image_path)
if image is None:
    print("Xato: rasm topilmadi:", image_path)
    raise SystemExit(1)

face = detect_face(image)
if face is None:
    print("Yuz topilmadi.")
    raise SystemExit(1)

# 5 ta nuqtani ham ko'rsatamiz — ular qayerda ekanini ko'zda ko'rish foydali.
# range(5) — 0,1,2,3,4. Har bir nuqtaning x qiymati 4+2*i, y qiymati 5+2*i o'rinda turadi.
for i in range(5):
    point_x = int(face[4 + 2 * i])
    point_y = int(face[5 + 2 * i])
    # Kichik qizil nuqta. Rang BGR: (0, 0, 255) — qizil. -1 — doira ichini to'ldirish.
    cv2.circle(image, (point_x, point_y), 4, (0, 0, 255), -1)

right_cheek, left_cheek, radius = find_cheeks(face)

print("O'ng yonoq markazi (rasmda chapda):", right_cheek)
print("Chap yonoq markazi (rasmda o'ngda):", left_cheek)
print("Doira radiusi:", radius, "piksel")

# Yonoqlar atrofiga yashil doira chizamiz. 2 — chiziq qalinligi.
cv2.circle(image, right_cheek, radius, (0, 255, 0), 2)
cv2.circle(image, left_cheek, radius, (0, 255, 0), 2)

cv2.imwrite(output_path, image)
print("Natija saqlandi:", output_path)
