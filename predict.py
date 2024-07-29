import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def load_and_preprocess_image(img_path, target_size):
    """Load and preprocess the image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def load_model_from_file(model_path):
    """Load the pre-trained model from the specified path."""
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def make_predictions(model, img_array):
    """Make predictions using the loaded model."""
    predictions = model.predict(img_array)
    return predictions

def get_class_names(train_dir):
    """Get class names from training directory."""
    class_names = os.listdir(train_dir)  # Assuming subfolder names are the class labels
    class_names.sort()  # Ensure consistent ordering
    return class_names

def main(model_path, img_path, train_dir):
    # Define target image size based on model requirements
    target_size = (224, 224)  # Adjust if needed

    # Load the model
    model = load_model_from_file(model_path)

    # Get class names from train directory
    class_names = get_class_names(train_dir)

    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path, target_size)

    # Make predictions
    predictions = make_predictions(model, img_array)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_label_index]
    probability_score = predictions[0][predicted_label_index]

    print(f"Predicted label: {predicted_label}, Probability: {probability_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pre-trained model and make a prediction on a new image")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to be predicted')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training dataset for inferring class names')

    args = parser.parse_args()
    main(args.model_path, args.img_path, args.train_dir)
