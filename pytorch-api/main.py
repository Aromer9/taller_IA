from app import app
from utils import get_prediction
from flask import Flask, jsonify, request


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        print("get prediction")
        print(get_prediction(image_bytes=img_bytes))

        score, class_info = get_prediction(image_bytes=img_bytes)
        class_id, class_name = class_info
        return jsonify({'class_id': class_id, 'class_name': class_name, 'score': round(score*100, 1)})

if __name__ == "__main__":
    app.run(port=5001)