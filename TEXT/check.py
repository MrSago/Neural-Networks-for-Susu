from keras.datasets import imdb
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

maxlen = 500
model_path = 'model.keras'
model = load_model(model_path)
# Получаем файл индекса слов, сопоставляя слова с индексами
word_index = imdb.get_word_index()

# Функция предсказания отзыва
def predict_sentiment(review):
    # Токенизация отзыва и приведение к нижнему регистру
    tokens = [word_index[word] if word in word_index else 0 for word in review.lower().split()]

    # Добавление паддинга
    tokens_padded = pad_sequences([tokens], maxlen=maxlen)

    # Предсказание
    prediction = model.predict(tokens_padded)
    return prediction[0][0]

# Пример использования функции
new_review = "This movie was excellent!"
prediction = predict_sentiment(new_review)
print(f"Review: {new_review}")
print(f"Review sentiment (0=negative, 1=positive): {prediction}")

new_review = "The movie was terrible and the acting was bad."
prediction = predict_sentiment(new_review)
print(f"Review: {new_review}")
print(f"Review sentiment (0=negative, 1=positive): {prediction}")
