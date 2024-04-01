import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from keras.models import load_model


def load_test_case(path):
    image = Image.open(path)

    # Конвертация изображения в оттенки серого, ресайзинг с помощью LANCZOS
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Инвертация изображения
    image = ImageOps.invert(image)
    image_array = np.array(image)
    image_array = image_array / 255.0

    # Добавить канал измерения
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    # Метка теста - имя изображения без расширения
    filename = os.path.splitext(os.path.basename(path))[0]
    test_label = np.array([int(filename)])

    return (image_array, test_label)


test_images = []
test_labels = []

# Загружаем 10 тестовых изображений (цифры от 0 до 9)
for i in range(10):
    image_path = f'./test/{i}.png'
    image_array, test_label = load_test_case(image_path)
    test_images.append(image_array)
    test_labels.append(test_label)

# Конвертируем список в numpy массив
test_images = np.vstack(test_images)  # Выполняем вертикальную конкатенацию
test_labels = np.concatenate(test_labels)  # Выполняем горизонтальную конкатенацию

# Загрузка модели
model_path = 'mnist_cnn_model.keras'
model = load_model(model_path)

# Оценка модели по тестовым данным
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

# Выбор случайного изображения
index = np.random.randint(0, len(test_images))
selected_image = test_images[index:index+1]
selected_label = test_labels[index]

# Предсказание класса изображения
predicted = model.predict(selected_image)
predicted_class = np.argmax(predicted, axis=-1)

string_result = \
    f"Predicted class: {predicted_class[0]}\n" +\
    f"True Label: {selected_label}"
print(string_result)

# Визуализация выбранного изображения
plt.imshow(selected_image.squeeze(), cmap=plt.cm.binary)
plt.title(string_result)
plt.show()
