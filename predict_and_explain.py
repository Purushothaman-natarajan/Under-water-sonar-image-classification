import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from explainer import generate_lime_mask, generate_splime_mask
from PIL import Image
import argparse

# Function to load model details
def load_model_details(model_path):
    # Detect the format and load the model accordingly
    if model_path.endswith('.keras'):
        print("Loading .keras format model...")
        model = tf.keras.models.load_model(model_path, compile=False)
    elif model_path.endswith('.h5'):
        print("Loading .h5 format model...")
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        print("Loading SavedModel using TFSMLayer...")
        model = tf.keras.Sequential([
            tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        ])

    # Get the target size dynamically
    input_shape = model.input_shape[1:3]

    return model, input_shape

# Function to load label encoder
def load_label_encoder(train_directory):
    labels = sorted(os.listdir(train_directory))
    label_encoder = {i: label for i, label in enumerate(labels)}
    return label_encoder

# Function to get image array
def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# Global counter for image naming
image_counter = 0

# Function to classify image and generate explanations
def classify_image_and_explain(image_path, model_path, train_directory, explanation_method, num_samples, num_features, segmentation_alg, kernel_size, max_dist, ratio):
    global image_counter
    image_counter += 1
    model, target_size = load_model_details(model_path)
    
    image = Image.open(image_path).resize(target_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    array = img_to_array(image)
    img_array = array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    label_encoder = load_label_encoder(train_directory)
    
    preds = model.predict(img_array)
    top_prediction = np.argmax(preds[0])
    top_label = label_encoder[top_prediction]
    top_prob = preds[0][top_prediction]

    if explanation_method == "lime":
        lime_mask, explanation_instance = generate_lime_mask(img_array[0], model, num_samples, num_features)
    elif explanation_method == "splime":
        explanation_mask = generate_splime_mask(img_array[0], model, num_features, num_samples, segmentation_alg, kernel_size, max_dist, ratio)
    else:
        raise ValueError("Invalid explanation method. Choose 'lime' or 'splime'.")

    if not os.path.exists("explanation"):
        os.makedirs("explanation")

    explanation_image = array_to_img(lime_mask.astype(np.uint8))
    explanation_image.save(f"explanation/{explanation_method}_explanation_{image_counter}.jpg")

    print(f"Predicted Label: {top_label}")
    print(f"Probability: {top_prob:.4f}")

    return explanation_image, top_label, top_prob

# Main function to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image and generate explanations using LIME or SP-LIME.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model (e.g., /path/to/model.h5 or /path/to/model.keras)")
    parser.add_argument("--train_directory", type=str, required=True, help="Path to the training directory (e.g., /path/to/train)")
    parser.add_argument("--explanation_method", type=str, choices=['lime', 'splime'], required=True, help="Explanation method to use (lime or splime)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for LIME")
    parser.add_argument("--num_features", type=int, default=10, help="Number of features for LIME")
    parser.add_argument("--segmentation_alg", type=str, choices=['quickshift', 'slic'], default='quickshift', help="Segmentation algorithm for SP-LIME")
    parser.add_argument("--kernel_size", type=int, default=2, help="Kernel size for segmentation algorithm")
    parser.add_argument("--max_dist", type=int, default=200, help="Max distance for segmentation algorithm")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio for segmentation algorithm")

    args = parser.parse_args()

    classify_image_and_explain(
        args.image_path,
        args.model_path,
        args.train_directory,
        args.explanation_method,
        args.num_samples,
        args.num_features,
        args.segmentation_alg,
        args.kernel_size,
        args.max_dist,
        args.ratio
    )
