import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_and_preprocess_image(img_path, target_size):
    """Load and preprocess the image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def load_all_models(model_dir):
    """Load all models from the specified directory."""
    models = {}
    for file_name in os.listdir(model_dir):
        if file_name.endswith('_model.keras'):
            model_path = os.path.join(model_dir, file_name)
            model_name = file_name.split('_model.keras')[0]  # Extract model name
            model = load_model(model_path)
            models[model_name] = model
            print(f"Model loaded from {model_path}")
    if not models:
        raise FileNotFoundError(f"No model files found in {model_dir}.")
    return models

def load_model_from_file(model_path):
    """Load a single model from the specified path."""
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

def compute_confusion_matrix_and_report(true_labels, predicted_labels, class_names, log_dir, model_name):
    """Compute confusion matrix and classification report, and save to log directory."""
    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    report = classification_report(true_labels, predicted_labels, target_names=class_names)

    # Print the classification report
    print(f"Model: {model_name}")
    print(report)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Save plot
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    conf_matrix_plot_file = os.path.join(log_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(conf_matrix_plot_file)
    plt.close()

    # Save results to log directory
    conf_matrix_file = os.path.join(log_dir, f'confusion_matrix_{model_name}.txt')
    report_file = os.path.join(log_dir, f'classification_report_{model_name}.txt')

    np.savetxt(conf_matrix_file, conf_matrix, fmt='%d', delimiter=',', header=','.join(class_names))
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"Confusion matrix and classification report saved to {log_dir} with model name: {model_name}")

def main(model_path, model_dir, img_path, test_dir, train_dir, log_dir):
    # Define target image size based on model requirements
    target_size = (224, 224)  # Adjust if needed

    if model_path:
        # Load a single model
        model = load_model_from_file(model_path)
        models = {os.path.basename(model_path): model}
    elif model_dir:
        # Load all models from a directory
        models = load_all_models(model_dir)
    else:
        raise ValueError("Either --model_path or --model_dir must be provided.")
    
    # Get class names from train directory
    class_names = get_class_names(train_dir)
    num_classes = len(class_names)

    # If an image path is provided, perform prediction on that image
    if img_path:
        img_array = load_and_preprocess_image(img_path, target_size)
        for model_name, model in models.items():
            print(f"Model: {model_name}")
            predictions = make_predictions(model, img_array)
            predicted_label_index = np.argmax(predictions, axis=1)[0]
            if predicted_label_index >= num_classes:
                raise ValueError(f"Predicted label index {predicted_label_index} is out of range for class names list.")
            predicted_label = class_names[predicted_label_index]
            probability_score = predictions[0][predicted_label_index]
            print('-'*20)
            print(f"Predicted label: {predicted_label}, Probability: {probability_score:.4f}")
            print('-'*20)

    # If a test directory is provided, perform batch predictions and evaluation
    if test_dir:
        files = [os.path.join(root, file) for root, _, files in os.walk(test_dir) for file in files if file.endswith(('png', 'jpg', 'jpeg'))]
        
        for model_name, model in models.items():
            true_labels = []
            predicted_labels = []

            for img_path in tqdm(files, desc=f"Processing images with {model_name}"):
                img_array = load_and_preprocess_image(img_path, target_size)
                predictions = make_predictions(model, img_array)
                predicted_label_index = np.argmax(predictions, axis=1)[0]
                if predicted_label_index >= num_classes:
                    raise ValueError(f"Predicted label index {predicted_label_index} is out of range for class names list.")
                predicted_label = class_names[predicted_label_index]
                
                true_label = os.path.basename(os.path.dirname(img_path))  # Assuming folder name is the label
                if true_label not in class_names:
                    raise ValueError(f"True label {true_label} is not in class names list.")
                
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
            
            # Compute and save confusion matrix and classification report
            compute_confusion_matrix_and_report(true_labels, predicted_labels, class_names, log_dir, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models and make predictions on new images or a test dataset")
    parser.add_argument('--model_path', type=str, help='Path to a single saved model')
    parser.add_argument('--model_dir', type=str, help='Directory containing saved models (loads all models in the folder)')
    parser.add_argument('--img_path', type=str, help='Path to the image to be predicted')
    parser.add_argument('--test_dir', type=str, help='Directory containing test dataset for batch predictions')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training dataset for inferring class names')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save prediction results')

    args = parser.parse_args()
    main(args.model_path, args.model_dir, args.img_path, args.test_dir, args.train_dir, args.log_dir)
