# exemplo com o classificadores CNN com Keras - sinal tratado como imagem
# Banco de dados de epilepsia

import scipy.io as sio
import numpy as np

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Input
sinal_ = sio.loadmat('sinal')
sinal_ = sinal_.get('sinal', 0)
#cria uma matriz
sinal = np.zeros(shape = (sinal_.shape[0], 1, sinal_.shape[1], 1))
#preenche a matriz com os valores desejados
for k in range(0, sinal_.shape[0]):
    sinal[k, 0, :, 0] = sinal_[k, :]
      
quantidade_exemplos = sinal.shape[0]
quantidade_amostras = sinal.shape[2]

#Label
label_ = sio.loadmat('label')
label_ = label_.get('label', 0)
#cria uma matriz
label = np.zeros(shape = (label_.shape[0], ))
#preenche a matriz com os valores desejados
for k in range(0, label_.shape[0]):
    label[k] = label_[k, 0] - 1
label = label.astype(np.uint8) 

from keras.utils import to_categorical
label= to_categorical(label, num_classes=None)

print(sinal.shape)
print(label.shape)

# Separa treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(sinal,label, test_size=0.2)

# definir arquitetura da CNN com Keras

from keras import models
model = models.Sequential()
model.add(layers.Conv2D(10, (1, 5), activation='relu', input_shape=(1, quantidade_amostras, 1)))
model.add(layers.MaxPooling2D((1, 2)))
model.add(layers.Conv2D(10, (1, 5), activation='relu'))
model.add(layers.MaxPooling2D((1, 2)))
model.add(layers.Conv2D(10, (1, 5), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
model.fit(X_train,  y_train, epochs=50)

val_loss, val_acc = model.evaluate(X_val, y_val)



