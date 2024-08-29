import logging
import os
import numpy as np
import cv2
import torch
import requests
from flask import Flask, jsonify
from urllib.parse import unquote


from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

app = Flask(__name__)
_logger = logging.getLogger("inference")

class Args:
    def __init__(self, device, draw, detector_weights, checkpoint, with_persons, disable_faces):
        self.device = device
        self.draw = draw
        self.detector_weights = detector_weights
        self.checkpoint = checkpoint
        self.with_persons = with_persons
        self.disable_faces = disable_faces

def setup_predictor(device='cpu'):
    setup_default_logging()
    args = Args(
        device=device,
        draw=False,
        detector_weights='models/yolov8x_person_face.pt',
        checkpoint='models/mivolo_imbd.pth.tar',
        with_persons=False,
        disable_faces=False
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.benchmark = True
    
    predictor = Predictor(args, verbose=True)
    return predictor

predictor = setup_predictor()

def url_to_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image from {url}")

    image_data = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to convert image from {url} to np array")

    return image


@app.route('/age-estimate/<path:image_url>', methods=['GET'])
def age_estimate(image_url):
    try:
        image_url = unquote(image_url)
        img = url_to_image(image_url)
        detected_objects, _ = predictor.recognize(img)
        age_estimated = detected_objects.ages[1]
        
        return jsonify({"age": age_estimated})

    except Exception as e:
        _logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=18751)