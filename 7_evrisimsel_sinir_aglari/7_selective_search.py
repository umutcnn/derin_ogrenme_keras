# eger  ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#         ^^^^^^^^^^^^^^^^^^^^^^^^^ hattasını alıyorsan 
#pip install --force-reinstall opencv-contrib-python
#pip install --no-cache --force-reinstall opencv-contrib-python
#cmd de bu satirları çalistir
import cv2
import random

image = cv2.imread("pyramid.jpg")
image = cv2.resize(image, dsize=(600, 600))
cv2.imshow("image", image)

# seçmeli aramayi baslatiyoruz
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#gorselimizi veriyoruz
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("Start")
rects = ss.process()

output = image.copy()

# sevilen bölgelerini renklendir ve görsellestiriyoruz
for (x, y, w, h) in rects[:50]:
    color = [random.randint(0, 255) for _ in range(3)]
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
