from flask import Flask, request
from experiments.CNN import *

app = Flask(__name__)

# 初始化模型、损失函数和优化器
model = CNN().to(device)


@app.route('/train', methods=['get', 'post'])
def train():
    train_model(model)
    return 'success'


@app.route('/predict', methods=['post'])
def predict():
    image = request.files['image']
    prediction = model_predict(model, image)
    prediction = round(prediction, 4)
    return str(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=False, processes=10)
