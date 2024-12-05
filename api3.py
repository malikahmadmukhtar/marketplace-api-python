import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
import cv2
from skimage.feature import local_binary_pattern
import pickle

app = Flask(__name__)

# Folder to store images
IMAGES_FOLDER = "stored_images"
FEATURES_FILE = "image_features.pkl"

os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Transformation for deep feature extraction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# In-memory cache for features
image_features = {}


def extract_deep_features(image_path):
    """Extract feature vector from an image using ResNet50."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image_tensor).squeeze(0).numpy()
    return features / np.linalg.norm(features)  # Normalize the feature vector


def extract_color_histogram(image_path):
    """Extract color histogram features."""
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_texture_features(image_path):
    """Extract texture features using Local Binary Patterns (LBP)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')  # Uniform LBP
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist


def load_features():
    """Load precomputed features from the file if available."""
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


def save_features():
    """Save features to the file."""
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(image_features, f)


@app.route('/recognize', methods=['POST'])
def recognize_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Save the input image temporarily
    image_file = request.files['image']
    input_image_path = os.path.join("temp_input.jpg")
    image_file.save(input_image_path)

    try:
        # Extract features for the input image
        input_deep_features = extract_deep_features(input_image_path)
        input_color_histogram = extract_color_histogram(input_image_path)
        input_texture_features = extract_texture_features(input_image_path)

        # Iterate through cached features and compute similarity
        closest_match = None
        highest_similarity = -1

        for filename, stored_features in image_features.items():
            deep_similarity = np.dot(input_deep_features, stored_features['deep'])
            color_similarity = cv2.compareHist(input_color_histogram, stored_features['color'], cv2.HISTCMP_CORREL)
            texture_similarity = np.dot(input_texture_features, stored_features['texture'])

            # Hybrid similarity (weighted sum)
            overall_similarity = (
                    0.6 * deep_similarity +
                    0.3 * color_similarity +
                    0.1 * texture_similarity
            )

            if overall_similarity > highest_similarity:
                highest_similarity = overall_similarity
                closest_match = filename

        if closest_match:
            return jsonify({
                'closest_match': closest_match,
                'similarity_score': float(highest_similarity)
            }), 200
        else:
            return jsonify({'message': 'No matching images found'}), 404

    finally:
        if os.path.exists(input_image_path):
            os.remove(input_image_path)


@app.route('/add-image', methods=['POST'])
def add_image():
    if 'image' not in request.files or 'filename' not in request.form:
        return jsonify({'error': 'Image file and filename are required'}), 400

    image_file = request.files['image']
    filename = request.form['filename']
    save_path = os.path.join(IMAGES_FOLDER, filename)

    # Ensure no overwriting
    if os.path.exists(save_path):
        return jsonify({'error': 'A file with this name already exists'}), 400

    # Save the image and compute its features
    image_file.save(save_path)
    features = {
        'deep': extract_deep_features(save_path),
        'color': extract_color_histogram(save_path),
        'texture': extract_texture_features(save_path),
    }

    # Update cache and save features to disk
    image_features[filename] = features
    save_features()

    return jsonify({'message': f'Image {filename} added successfully'}), 200


if __name__ == '__main__':
    # Load cached features on server start
    image_features = load_features()
    app.run(debug=True)
