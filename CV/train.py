# Иморт нужных библиотек
from time import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Загрузка и подготовка MNIST датасета
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализовать значения пикселей в диапазоне от 0 до 1.
train_images, test_images = train_images / 255.0, test_images / 255.0

# Изменение формы изображений в соответствии с ожиданиями сверточного слоя
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Построение сверточной базы, используя стек слоев Conv2D и MaxPooling2D
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Добавление плотного слоя чтобы выполнить классификацию
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10-классов множества цифр

# Компиляция и обучение модели
start_time = time()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=32, epochs=5,
          validation_data=(test_images, test_labels))
end_time = time()

model.summary()
print(f'Training time: {end_time - start_time} seconds')

# Оценка модели
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

# Сохранение модели (опционально)
model.save('mnist_cnn_model.keras')
