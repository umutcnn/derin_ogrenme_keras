#gerekli kutuphaneleri import ediyoruz
import numpy as np
import cv2
import os #dosya islemleri icin
from sklearn.model_selection import train_test_split # veri kümesini egitim ve test setlerine ayırmak icin kullaniyoruz
from sklearn.metrics import confusion_matrix #farklı metriklerle model performansini degerlendirmek icin kullanacagiz
import seaborn as sns #gorsellestirme kutuphanesi
import matplotlib.pyplot as plt #gorsellestirme kutuphanesi
from keras.models import Sequential #tabani olusturmak icin kullanacagiz
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization #sinir agi katmanlarini olusturmak şcşn kullanacagiz
from keras.utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator #data augmentation islemleri icin kullanacagiz
import pickle #nesneleri kaydetmek ve yuklemek icin kullanilir

path = "myData"

myList = os.listdir(path) #tum klasorlerimi aliyorum
noOfClasses = len(myList) #mylist uzunlugu

print("Label(sınıf) sayısı: ", noOfClasses)
# Label sınıf sayisi 0-9 arasinda 10 sayidir

images = []
classNo = []

#klasorler icerisinde geziyorum
for i in range(noOfClasses):
    myImageList = os.listdir(path+"\\"+str(i))

    #klasorlerin icerisindeki imgleri aliyorum
    for j in myImageList:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)# j resim adi
        img = cv2.resize(img,(32,32)) #egitecegimiz netural network girdisi 32x32 dir bu nedenle resize ediuoruz
        images.append(img)
        classNo.append(i)
        
print(len(images))
print(len(classNo))

images = np.array(images) #resimleri numpy array ceviriyorum
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)

# veriyi ayiriyoruz
#resim , klas, test_size %50 ye %50 ayiriyorum, random_state resimleri bolmek icin kullanilan parametredir
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size= 0.5, random_state = 42)
#dogrulama yapma icin bu satiri kullaniyoruz girdileri aynidir
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size= 0.2, random_state = 42)

print(images.shape)
print(x_train.shape) # eğitim için ayrılan
print(x_test.shape) # test için ayrılan
print(x_validation.shape) # doğrulama için ayrılan

"""
# vis veri içine bakıyoruz
fig, axes = plt.subplots(3,1,figsize=(7,7))
fig.subplots_adjust(hspace = 0.5)
sns.countplot(y_train, ax = axes[0])
axes[0].set_title("y_train")

sns.countplot(y_test, ax = axes[1])
axes[1].set_title("y_test")

sns.countplot(y_validation, ax = axes[2])
axes[2].set_title("y_validation")
"""


# on isleme adimlarina geciyoruz 
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #bgr to gray siyah beyaza cevirdim
    img = cv2.equalizeHist(img) # Histogramı 0-255 arasına kadar genişlettik
    img = img / 255 # normalize ettik
    
    return img

"""
idx = 1820
img = preProcess(x_train[idx])
img = cv2.resize(img,(300,300))
cv2.imshow("Preprocess", img)
"""

# tum verilerimize on isleme yapiyoruz
#map iki parametre alır 1.si hangi fonksiyondur ikincisi de kime uygulayacagidir
x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

#verilerimizi reshape yapiyoruz bunun nedeni verilerimizi egitime hazir hale getirmek icin yapiyoruz
x_train = x_train.reshape(-1,32,32,1)
# -1 in anlamı boyutlarımız 32,32 olsun da resim kac tane varsa ona göre ayarla 
print(x_train.shape)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)


# data generate yani verilerimizi arttirip genisletiyoruz
#width_shift_range = goruntunun genisligini %10 kadar yatayda rastgele olacak sekilde arttirip azaltabilirsin
#height_shift_range = goruntunun genisligini %10 kadar dikeyde rastgele olacak sekilde arttirip azaltabilirsin
#zoom_range = %10 olacak sekilde yakinlastirip uzaklastirabilirsin
#rotation_range = goruntuleri en fazla 10 derece olacak sekilde donderebilirsin
dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.1,
                             rotation_range = 10)

dataGen.fit(x_train)
# fit edelim x_train kullanarak yeni resimler üretelim


#verileri kategorik hale getir onehotencoder
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


model = Sequential()
#input_shape girdinin boyutu
#relu = Rectified Linear Activation genellikle cnn lerde bu kullanilir
# padding same bir sıra piksel ekliyor same ise bir sıra piksel ekleme anlamına gelir
model.add(Conv2D(input_shape = (32,32,1), filters = 8, kernel_size = (5,5), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

#benzer islemi bir kere daha yapiyorum
model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

# yeni veri ürettiğimiz için overfittingi (ezberlemeyi) engellemek için
# dropout ekliyoruz
model.add(Dropout(0.2))#%20 oraninda rastgele devre disi birak
model.add(Flatten())#duzlestirme
# 256 Hücre
model.add(Dense(units=256, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(units=noOfClasses, activation = "softmax")) # çıktımız

#modelimizi deremek icin
#loss = egitim sirasinda ne kadar hata yapildigini hesaplar
#optimizer = parametrelerimizi bulmamızı sağlayan
#metrics = accuracy  modelin sonuçlarını yüzde olarak değerlendirmemizi sağlar
model.compile(loss = "categorical_crossentropy", optimizer = ("Adam"), metrics=["accuracy"])


batch_size = 250

# validation_data= dogrulama icin kullanilicak veriler
# epochs=15 egirim veri kumesinin kaç kez calısacagini belirler
# steps_per_epoch = her epochs ta kac adim atilacagini belirler
# shuffle = epoch yapmadan onde karsilastirma yapilip yapilmayacagini belirler (1 de yapilir)
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size),
                                        validation_data = (x_validation, y_validation),
                                        epochs = 15, steps_per_epoch = x_train.shape[0]//batch_size,
                                        shuffle = 1)


pickle_out = open("model_trained_new.p","wb")
# pickle out modelemizi depolayacak
pickle.dump(model, pickle_out)
pickle_out.close()

# % değerlendirme yapiyoruz

hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Loss")
plt.plot(hist.history["val_loss"], label = "Val Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()
plt.show()

#modelin degerlendirilmesi
score = model.evaluate(x_test, y_test, verbose = 1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])


#dogrulama verileri ile karısıklık matrisini hesaplayacagiz
y_pred = model.predict(x_validation)

y_pred_class = np.argmax(y_pred, axis = 1)

Y_true = np.argmax(y_validation, axis = 1)

cm = confusion_matrix(Y_true, y_pred_class)

#karisiklik matrisinin isi haritasi ile gorsellestir
f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot=True, linewidths=0.01, cmap="Greens", linecolor = "gray", fmt = ".1f", ax = ax)
plt.xlabel("tahmin")#x 
plt.ylabel("gercek")#y
plt.title("Karışıklık matrisi")#karısıklık matrisi
plt.show()



