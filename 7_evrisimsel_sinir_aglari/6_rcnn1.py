from keras.applications.resnet50 import preprocess_input 
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

#daha once yazdigimiz kodlari import ediyoruz
from sliding_window_2 import sliding_window
from image_pyramid import image_pyramid
from non_max_supression_3 import non_max_suppression

WIDTH = 600
HEIGHT = 600
PYR_SCALE = 1.5 # image pyramid scale
WIN_STEP = 16 # sliding step size
ROI_SIZE = (200,150)
INPUT_SIZE = (224, 224) ## resnete sokacagimiz resmin input boyutu
# sabit parametreler hicbir zaman degismeyecek parametreler olacakları icin buyuk yazdık

print("Loading ResNet")
model = ResNet50(weights = "imagenet", include_top = True)


#nesne takibi yapacagimiz resim
original = cv2.imread("husky.jpg")
original = cv2.resize(original, dsize=(WIDTH, HEIGHT))
# once bizim fonksiyonlarımıza 600,600 seklinde sokuyoruz en son resnet'e
# 224,224 seklinde sokacagiz
cv2.imshow("Husky", original)

(H, W) = original.shape[:2]

# image pyramid
pyramid = image_pyramid(original, PYR_SCALE, ROI_SIZE) # In each iteration, we will run a sliding window.

rois = []
locs = []

for image in pyramid:

    #image_pyramid te bir PYR_SCALE uyguluyoruz bunu scaleye de uygulamalıyız cunku bir dengesizlik olusur
    scale = W/float(image.shape[1])
    
    #WIN_STEP piksel kayma
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        x = int(x*scale)
        y = int(y*scale)
        w = int(ROI_SIZE[0]*scale)
        h = int(ROI_SIZE[1]*scale)

        #roiOrig alıp sınıflandırmada kullanmam gerekiyor onun için resize ediyoruz
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        # preprocess_input kullanarak resnete hazir hale getirmis oluyorum
        roi = preprocess_input(roi)

        #rois icerisine roi ekliyorum
        rois.append(roi)
        locs.append((x,y,x+w,y+h))

rois = np.array(rois, dtype="float32")

print("classification")

preds = model.predict(rois)

preds = imagenet_utils.decode_predictions(preds, top=1)

labels = {}
#bu degerden yuksek tahminleri isleme alicam
min_conf = 0.9 # 0.95, 0.8

for (i, p) in enumerate(preds):

    # tahminler 3 tane değer dondurulecek 1. imageNet_id, 2. label(sınıf)
    # 3. yüzde tahmin değeri.
    (_, label, prob) = p[0]

    #eger prob, min_conf tan buyukse
    if prob >= min_conf:
        #ozzetle min_conf degerinden yuksek tahminleri isleme aliyoruz

        box = locs[i]
        #kutu ve olasilik
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

#ayiklama ve gorsellestirme yapiyoruz
for label in labels.keys():
    #orijinal resmi bozmamak icin copy aliyoruz
    clone = original.copy()

    #kutucuklari cizdiriyorum
    for (box, prob) in labels[label]:
        #ayikliyorum
        (startX, starY, endX, endY) = box
        #copy uzerinde cizim islemlerini yapiyoruz yesil kalinlik 2
        cv2.rectangle(clone, (startX, starY), (endX, endY), (0,255,0), 2)
    #gosteriyorum
    cv2.imshow("ilk", clone)


    clone = original.copy()

    # non-maxima labelleri boluyoruz (kutular ve olasiliklar olarak)
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])

    #kutucukların max olasiliği olani al
    boxes = non_max_suppression(boxes, proba)

    #sonra bunu ciz
    for (startX, starY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, starY), (endX, endY), (0,255,0), 2)
        y = starY - 10 if starY - 10 > 10 else starY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    
    cv2.imshow("max bulunan olasilikli eskimo_dog",clone)

k = cv2.waitKey(0) & 0xFF # -> esc tusunu al

if k == 27: # esc basinca cik
    cv2.destroyAllWindows()
    
    












