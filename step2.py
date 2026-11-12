# 2-QADAM: YuNet yordamida yuzni topish va atrofiga ramka chizish

import cv2

# Fayl nomlari: kiruvchi rasm, YuNet modeli va natija saqlanadigan rasm.
image_path = "selfie.jpg"
model_path = "face_detection_yunet_2023mar.onnx"
output_path = "step2_face.jpg"

# Rasmni o'qiymiz (1-qadamdagidek).
image = cv2.imread(image_path)
if image is None:
    print("Xato: rasm topilmadi:", image_path)
    raise SystemExit(1)

height, width, channels = image.shape

# YuNet yuz detektorini yaratamiz.
# Parametrlar: model fayli, qo'shimcha config (kerak emas, shuning uchun ""),
# rasm o'lchami (kenglik, balandlik) — detektor qaysi o'lchamdagi rasmni kutishini bilishi kerak.
detector = cv2.FaceDetectorYN.create(model_path, "", (width, height))

# Ishonch chegarasi: 0.9 — "90% ishonch bo'lsa, buni yuz deb hisobla".
# Pastroq qilsak, ko'proq (lekin ba'zan noto'g'ri) yuzlar topiladi.
detector.setScoreThreshold(0.9)

# Yuzlarni qidiramiz. detect() ikki narsa qaytaradi:
#   1) holat raqami (bizga kerak emas, shuning uchun "_" deb yozamiz)
#   2) topilgan yuzlar ro'yxati (yuz topilmasa — None)
_, faces = detector.detect(image)

if faces is None:
    print("Yuz topilmadi. Boshqa, yorug'roq rasm bilan urinib ko'ring.")
    raise SystemExit(1)

print("Topilgan yuzlar soni:", len(faces))

# Bir nechta yuz bo'lsa, birinchisini olamiz.
face = faces[0]

# Har bir yuz — 15 ta sondan iborat qator:
#   [0..3]  — ramka: x, y (chap-yuqori burchak), kenglik, balandlik
#   [4..13] — 5 ta nuqta (landmark): ko'zlar, burun, og'iz burchaklari (3-qadamda ishlatamiz)
#   [14]    — ishonch darajasi (0 dan 1 gacha)
# Piksel koordinatasi butun son bo'lishi kerak, shuning uchun int() qilamiz.
x = int(face[0])
y = int(face[1])
w = int(face[2])
h = int(face[3])
score = float(face[14])

print("Yuz ramkasi: x =", x, ", y =", y, ", kenglik =", w, ", balandlik =", h)
print("Ishonch darajasi:", round(score, 3))

# Yuz atrofiga yashil to'rtburchak chizamiz.
# Rang BGR tartibida: (0, 255, 0) — yashil. Oxirgi son — chiziq qalinligi.
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Natijani yangi faylga saqlaymiz (asl selfie.jpg o'zgarmaydi).
cv2.imwrite(output_path, image)
print("Natija saqlandi:", output_path)
