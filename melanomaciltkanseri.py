import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix



# Rastgele sonuclar uretmesini engeller
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Veri Yollarini Belirttik
egitim_dir= 'C:\\Users\\Sila Kara韩\\OneDrive\\Masaüstü\\dataset\\train'  
dogrulama_dir= 'C:\\Users\\Sila Kara韩\\OneDrive\\Masaüstü\\dataset\\validation'  

# Goruntulerin boyutlandirmasi ve normalizasyon islemlerini yaptik
egitim_datagen= ImageDataGenerator(rescale=1./255,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

dogrulama_datagen= ImageDataGenerator(rescale=1./255)


dogrulama_generator = dogrulama_datagen.flow_from_directory(dogrulama_dir,
                                                target_size=(150, 150),
                                                batch_size=38,
                                                class_mode='binary')

egitim_generator = egitim_datagen.flow_from_directory(egitim_dir,
                                                      target_size=(150, 150),
                                                      batch_size=38,
                                                      class_mode='binary') 

# GoogleNet modelini oluşturma
def create_googlenet_modeli(input_shape):
    model=Sequential()
    
    
    model.add(Conv2D(64,(7,7),strides=(2,2),padding='same',activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(64,(1,1), padding='same', activation='relu'))
    model.add(Conv2D(192,(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))

    model.add (Conv2D(128,(1,1),padding='same',activation='relu'))
    model.add (Conv2D(128,(1,1),padding='same',activation='relu'))
    model.add (Conv2D(128,(1,1),padding='same',activation='relu'))
    model.add (Conv2D(128,(1,1),padding='same',activation='relu'))
 
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    return model

model = create_googlenet_modeli(input_shape=(150,150,3))
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history=model.fit(egitim_generator,
                 steps_per_epoch=100,
                 epochs=20,
                 validation_data=dogrulama_generator,
                 validation_steps=50)


dogruluk= history.history['accuracy']
dogrulama_dogruluk=history.history['val_accuracy']
kayip=history.history['loss']
dogrulama_kayibi=history.history['val_loss']
epoch_araligi=range(20)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epoch_araligi,dogruluk,label='Egitim Dogrulugu')
plt.plot(epoch_araligi,dogrulama_dogruluk,label='Dogrulama Dogrulugu')
plt.legend(loc='lower right')
plt.title('Egitim ve Dogrulama Dogrulugu')
plt.subplot(1,2,2)
plt.plot(epoch_araligi,kayip,label='Egitim Kaybi')
plt.plot(epoch_araligi,dogrulama_kayibi,label='Dogrulama Kaybi')
plt.legend(loc='upper right')
plt.title('Egitim ve Dogrulama Kaybi')
plt.show()

# Confusing Matrix
dogrulama_generator.reset()
y_pred= model.predict(dogrulama_generator,steps=len(dogrulama_generator))
y_pred_classes= (y_pred > 0.5).astype("int32")
y_true = dogrulama_generator.classes

cm = confusion_matrix(y_true,y_pred_classes)
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (FN+FP+TP)

print("Karmasiklik Matrixi: ")
print(cm)
for i in range(len(TP)):
    print(f"\n Sinif {i}:")
    print(f" True Positive (TP): {TP[i]}")
    print(f" False Positive (FP): {FP[i]}")
    print(f" False Negative (FN): {FN[i]}")
    print(f" True Negative (TN): {TN[i]}")