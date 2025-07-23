# brightness_control.py
# 
# Hak Cipta (c) 2025 ARIEL APRIELYULLAH
# 
# Dengan ini diberikan izin, secara gratis, kepada siapa pun yang memperoleh salinan
# perangkat lunak ini dan file dokumentasi terkait ("Perangkat Lunak"), untuk menggunakan
# Perangkat Lunak tanpa batasan, termasuk namun tidak terbatas pada hak untuk menyalin,
# mengubah, menggabungkan, menerbitkan, mendistribusikan, melisensikan ulang, dan/atau
# menjual salinan Perangkat Lunak, serta mengizinkan orang kepada siapa Perangkat Lunak
# ini diberikan untuk melakukan hal yang sama, sesuai dengan ketentuan berikut:
# 
# Pemberitahuan hak cipta di atas dan pemberitahuan izin ini harus disertakan dalam semua
# salinan atau bagian penting dari Perangkat Lunak.
# 
# PERANGKAT LUNAK INI DISEDIAKAN "SEBAGAIMANA ADANYA", TANPA JAMINAN APA PUN, BAIK TERSURAT
# MAUPUN TERSIRAT, TERMASUK NAMUN TIDAK TERBATAS PADA JAMINAN DIPERDAGANGKAN, KESESUAIAN UNTUK
# TUJUAN TERTENTU, DAN TIDAK MELANGGAR. DALAM HAL APA PUN PENULIS ATAU PEMEGANG HAK CIPTA
# TIDAK BERTANGGUNG JAWAB ATAS KLAIM, KERUSAKAN, ATAU KEWAJIBAN LAINNYA YANG TIMBUL DARI,
# DARI, ATAU BERKAITAN DENGAN PERANGKAT LUNAK ATAU PENGGUNAAN ATAU HAL-HAL LAIN YANG TERKAIT
# DENGAN PERANGKAT LUNAK INI.


import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

# Inisialisasi kamera dan Mediapipe
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break  # Jika gagal membaca frame, keluar dari loop

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []

    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        # Koordinat jari jempol (id=4) dan telunjuk (id=8)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Gambar lingkaran dan garis
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Hitung jarak antara jempol dan telunjuk
        length = hypot(x2 - x1, y2 - y1)

        # Interpolasi jarak ke tingkat kecerahan
        bright = np.interp(length, [15, 220], [0, 100])
        print(bright, length)

        # Atur tingkat kecerahan layar
        sbc.set_brightness(int(bright))

    # Tampilkan frame
    cv2.imshow('Image', img)

    # Tombol untuk keluar dari aplikasi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepas resource kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
