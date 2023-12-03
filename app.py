from flask import Flask, render_template, request
import numpy as np
import os
import cv2

from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

app = Flask('Currency Detection Using SIFT & HOG')

# Load the trained models
svm_model = joblib.load('svm_model.pkl')
knn_model = joblib.load('knn_model.pkl')


# Function to extract SIFT features from an image
def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors.flatten() if descriptors is not None else None


# Function to extract HOG features from an image
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
    # Enhance the contrast of the HOG image (optional)
    hog_image_rescaled = exposure.rescale_intensity(_, in_range=(0, 10))
    return features, hog_image_rescaled


# Function to resize and pad or truncate features to a fixed size
def resize_and_pad_features(features, target_size=100):
    if len(features) >= target_size:
        return features[:target_size]
    else:
        pad_size = target_size - len(features)
        return np.pad(features, (0, pad_size), mode='constant', constant_values=0)


# Function to classify an image using SVM and KNN models
def classify_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Adjust target size as needed

    # Extract SIFT features
    sift_features = extract_sift_features(image)
    sift_features = resize_and_pad_features(sift_features)

    # Extract HOG features
    hog_features, _ = extract_hog_features(image)
    hog_features = resize_and_pad_features(hog_features)

    # Combine features for classification
    combined_features = np.concatenate([sift_features, hog_features])

    # Reshape features to match the training data
    combined_features = combined_features.reshape(1, -1)

    # Predict using SVM model
    svm_prediction = svm_model.predict(combined_features)[0]

    # Predict using KNN model
    knn_prediction = knn_model.predict(combined_features)[0]

    return svm_prediction, knn_prediction


@app.route("/", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            # Saving the uploaded image temporarily
            image_path = os.path.join("static", "uploaded_image.jpg")
            image_file.save(image_path)

        

            # Classify the uploaded image
            svm_prediction, knn_prediction = classify_image(image_path)    

            return render_template("index.html", class_name=svm_prediction)
        
    else:
        return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)