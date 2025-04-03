from flask import Flask, jsonify, request
import random
import os
from keyword_spotting_service import Keyword_Spotting_Service, start_service

"""
client -> POST request -> server -> prediction back to client

"""

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # get audio file
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)
    # start service
    service = start_service()

    # make prediction
    predicted_keyword = service.predict(file_name)

    # remove audio file
    os.remove(file_name)

    # send back predicted keyword in json format
    data = {"keyword": predicted_keyword}

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)
