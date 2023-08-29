import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

#uyarilari kapatiyoruz
import warnings
warnings.filterwarnings("ignore")

#resimleri su uzantidan al sondaki *.png ismi ne olursa olsun sonu png olanlari al demek
imgs = glob.glob("./img_nihai/*.png")

ArithmeticError(
)

width = 125
height = 50

X = []
Y = []

#resimlerin icerisinde dolasiyorum
for img in imgs:
    
    #img al
    filename = os.path.basename(img)

    #resmin adini _ cizgiye gore ayiriyorum boylelikle hangi yone hareket etmem gerektigini buluyorum
    label = filename.split("_")[0]

    #resmin boyutunu degistiriyorum
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    #sonra resize ettiğim resmi normalize ediyorum
    im = im / 255
    #resimleri ekliyorum
    X.append(im)
    #etiketleri(yonu) ekliyorum
    Y.append(label)

#array a ceviriyorum
X = np.array(X)
# kac resim oldugunu, yukseklik, genislik, chanelle degeri yani siyah beyaz kullaniyorum
X = X.reshape(X.shape[0], width, height, 1)

#y den kac tane var ona bakiyorum
# sns.countplot(Y)

#
def onehot_labels(values):
    label_encoder = LabelEncoder()

    # verilen degerleri tamsayı olarak kodluyoruz
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


Y = onehot_labels(Y)
# x= resim, y= etiketler, test_size=  test verilerimiz %25 geriye kalanlar egitim icin kullanicak, rastgele bir bolunme olustur
train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)    

# cnn model
model = Sequential() 

# 32 adet 3x3 boyutunda filtre içeren convolutional layer ekliyoruz,
# relu aktivasyon fonksiyonu kullaniyoruz ve giriş şekli (width, height, chanelle= 1) olsun
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))

# 64 adet 3x3 boyutunda filtre içeren Convolutional layer ekle,
# relu aktivasyon fonksiyonu kullan
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))

# 2x2 boyutunda Max Pooling layer(piksel ekliyoruz) ekle
model.add(MaxPooling2D(pool_size = (2,2)))

# %25 dropout uygula (seyreltme uyguluyoruz)
model.add(Dropout(0.25))

# katmanlari düzlestiriyoruz
model.add(Flatten())

# 128 noronlu Fully Connected (Dense) layer ekle, relu aktivasyon fonksiyonu kullan
model.add(Dense(128, activation = "relu"))

# %40 dropout uygula
model.add(Dropout(0.4))

# 3 noronlu Fully Connected (Dense) layer ekle, Softmax aktivasyon fonksiyonu kullan
model.add(Dense(3, activation = "softmax"))

# if os.path.exists("./trex_weight.h5"):
#     model.load_weights("trex_weight.h5")
#     print("Weights yuklendi")    

# modelimizi derliyoruz
model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

# modeli egit (epochs= resimlerin kac kere egitilecegi yazilir,batch_size= resimlerin kac grup halinde iterasyona sokacagimizi belirliyoruz )
model.fit(train_X, train_y, epochs = 35, batch_size = 64)

#sonuclari aliyoruz
score_train = model.evaluate(train_X, train_y)
print("Eğitim doğruluğu: %",score_train[1]*100)    
    
score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %",score_test[1]*100)      
    
 
open("model_news.json","w").write(model.to_json())
model.save_weights("trex_weight_news.h5")   
    
    
    