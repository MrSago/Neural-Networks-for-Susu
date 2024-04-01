# Иморт нужных библиотек
from time import time
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences

# Загрузка датасета IMDb
num_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Предварительная обработка данных дополнив паддингами до длины максимальной последовательности
maxlen = 500
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Построение архитектуры нейросети
model = Sequential()
model.add(Embedding(num_words, 32, input_length=maxlen))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Компиляция и обучение модели
start_time = time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))
stop_time = time()

model.summary()
print(f'Training time: {stop_time - start_time} seconds')

# Оценка модели на тестовых данных
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: %.2f%%" % (scores[1] * 100))

# Сохранение модели (опционально)
model.save('model.keras')
