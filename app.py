import pickle
import numpy as np
from flask import Flask, request


model = None
app = Flask(__name__)


def load_model():
    global model


# model относится к глобальной переменной
with open('iris_trained_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Работает лишь в случае, если образец один
    if request.method == 'POST':
        data = request.get_json() # Получает отправляемые данные в формате json
        data = np.array(data)[np.newaxis, :] # преобразует фигуру из (4,) в (1, 4)
        prediction = model.predict(data) # применяет к данным глобально загруженную модель
    return str(prediction[0])


if __name__ == '__main__':
    load_model() # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
