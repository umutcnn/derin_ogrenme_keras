import cv2
import pickle
import numpy as np


# on isleme adimlarina geciyoruz 
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #bgr to gray siyah beyaza cevirdim
    img = cv2.equalizeHist(img) # Histogramı 0-255 arasına kadar genişlettik
    img = img / 255 # normalize ettik
    
    return img

cap = cv2.VideoCapture(0) #webcam ac
#genislik ve yuksekligi ayarliyorum
cap.set(3,480)
cap.set(4,480)

#model_trained_new dosyasını oku
pickle_in = open("model_trained_new.p","rb")
model = pickle.load(pickle_in)

#kamera aciksa 
while True:

    success, frame = cap.read()
    
    img = np.asarray(frame) #framemizi bir arraya ceviriyoruz
    img = cv2.resize(img, (32,32)) #resmimizin boyutunu ayarliyoruz
    img = preProcess(img) # on islemeye sokuyoruz
    img = img.reshape(1,32,32,1) # 1 tane resim 32,32 buyut sondaki bir ise renk kanalını 
    
    # tahmin edelim
    #resim uzerinde sınıf tahmini yapıyoruz sonucumuz float cikmasin diye intledik 
    prediction = model.predict(img)
    classIndex = np.argmax(prediction)  # En yüksek olasılığa sahip sınıfın indeksi
    probVal = np.amax(prediction)  # En yüksek olasılık değeri

    print(classIndex, probVal) #tahmin elen sinifi ve bu sifin olasiligini ekrana yazdiriyoruz
    
    #eger %40 uzerindeyse 
    if probVal > 0.4:
        #yazdır
        cv2.putText(frame, str(classIndex) + " " + str(probVal), (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        
        cv2.imshow("Rakam Siniflandirma", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"): break