# 1-QADAM: rasmni ochish va piksel rangini o'qish

# cv2 — OpenCV kutubxonasi. Rasmlar bilan ishlash uchun kerak.
import cv2

# Rasm fayli nomi. Fayl shu papkada turishi kerak.
image_path = "selfie.jpg"

# Rasmni o'qiymiz. Natija — raqamlardan iborat katta jadval (numpy massiv).
image = cv2.imread(image_path)

# Agar fayl topilmasa yoki buzilgan bo'lsa, imread None qaytaradi.
# Shuning uchun tekshiramiz va xato bo'lsa dasturni to'xtatamiz.
if image is None:
    print("Xato: rasm topilmadi yoki o'qib bo'lmadi:", image_path)
    raise SystemExit(1)

# image.shape uchta son beradi: (balandlik, kenglik, kanallar soni).
# Kanallar soni 3 — har bir pikselda 3 ta rang qiymati bor.
height, width, channels = image.shape

print("Rasm o'lchami:")
print("  Kenglik (width):  ", width, "piksel")
print("  Balandlik (height):", height, "piksel")
print("  Kanallar soni:     ", channels)

# Markaziy pikselning koordinatalarini topamiz.
# // — butun songa bo'lish (masalan, 7 // 2 = 3), chunki piksel raqami butun bo'lishi kerak.
center_x = width // 2
center_y = height // 2

# Pikselni olamiz. DIQQAT: OpenCV'da avval qator (y), keyin ustun (x) yoziladi.
pixel = image[center_y, center_x]

# OpenCV ranglarni BGR tartibida saqlaydi: Blue (ko'k), Green (yashil), Red (qizil).
# Ko'pchilik dasturlar RGB ishlatadi, OpenCV esa teskari — BGR.
blue = int(pixel[0])
green = int(pixel[1])
red = int(pixel[2])

print()
print("Markaziy piksel joyi: x =", center_x, ", y =", center_y)
print("BGR qiymati (OpenCV tartibi):", (blue, green, red))
print("RGB qiymati (oddiy tartib):  ", (red, green, blue))

# Har bir qiymat 0 dan 255 gacha: 0 — bu rang umuman yo'q, 255 — eng yorqin.
print()
print("Tushuntirish:")
print("  Blue  (ko'k)   =", blue)
print("  Green (yashil) =", green)
print("  Red   (qizil)  =", red)
