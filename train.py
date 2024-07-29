import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import (VGG16, VGG19, ResNet50, ResNet101, InceptionV3,
                                           DenseNet121, DenseNet201, MobileNetV2, Xception, InceptionResNetV2, 
                                           NASNetLarge, NASNetMobile, EfficientNetB0, EfficientNetB7)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def load_and_preprocess_image(filename, label, image_size):
    # Load image
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
    
    # Ensure the image tensor has shape
    if not tf.executing_eagerly():
        image.set_shape([None, None, 3])

    # Resize image to the specified size
    image = tf.image.resize(image, [image_size[0], image_size[1]])  # Use height and width from the tuple
    
    # Normalize image to [0, 1]
    image = image / 255.0
    image.set_shape([image_size[0], image_size[1], 3])

    return image, label

def create_dataset(data_dir, labels, image_size, batch_size):
    image_files = []
    image_labels = []

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir):
            image_files.append(os.path.join(label_dir, image_file))
            image_labels.append(label)

    # Create a mapping from labels to indices
    label_map = {label: idx for idx, label in enumerate(labels)}
    image_labels = [label_map[label] for label in image_labels]

    # Convert to TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((image_files, image_labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y, image_size))
    dataset = dataset.shuffle(buffer_size=len(image_files))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def create_and_train_model(base_model, model_name, shape, X_train, X_val, num_classes, labels, log_dir, model_dir,
                           epochs, optimizer_name, learning_rate, step_gamma, alpha, batch_size, patience):
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    predictions = Dense(num_classes, activation='softmax')(x)  # Use the number of classes
    model = Model(inputs=base_model.input, outputs=predictions)

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,  # Adjust this according to your needs
        decay_rate=step_gamma
    )

    # Select the optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = Adam(learning_rate=lr_schedule)
    elif optimizer_name.lower() == 'sgd':
        optimizer = SGD(learning_rate=lr_schedule, momentum=alpha)  # Example settings for SGD
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Set up callbacks
    checkpoint = ModelCheckpoint(os.path.join(model_dir, f'{model_name}_best_model.keras'), 
                                 monitor='val_accuracy', save_best_only=True, save_weights_only=False, 
                                 mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1)

    # Train the model
    history = model.fit(X_train, epochs=epochs, validation_data=X_val, batch_size=batch_size,
                        callbacks=[checkpoint, early_stopping])

    # Save training logs
    with open(os.path.join(log_dir, f'{model_name}_training.log'), 'w') as f:
        num_epochs = len(history.history['loss'])  # Get the actual number of epochs completed
        for epoch in range(num_epochs):
            f.write(f"Epoch {epoch + 1}, "
                    f"Train Loss: {history.history['loss'][epoch]:.4f}, "
                    f"Train Accuracy: {history.history['accuracy'][epoch]:.4f}, "
                    f"Val Loss: {history.history['val_loss'][epoch]:.4f}, "
                    f"Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}\n")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_val)
    print(f'Test Accuracy for {model_name}: {test_accuracy:.4f}')
    print(f'Test Loss for {model_name}: {test_loss:.4f}')

    # Optionally, save the trained model
    model.save(os.path.join(model_dir, f'{model_name}_final_model.keras'))

def main(base_model_names, shape, data_path, log_dir, model_dir, epochs, optimizer, learning_rate, step_gamma, alpha, batch_size, patience):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Extract labels from folder names
    labels = sorted([d for d in os.listdir(os.path.join(data_path, 'train')) if os.path.isdir(os.path.join(data_path, 'train', d))])
    num_classes = len(labels)

    # Load data
    X_train = create_dataset(os.path.join(data_path, 'train'), labels, shape, batch_size)
    X_val = create_dataset(os.path.join(data_path, 'val'), labels, shape, batch_size)

    if not base_model_names:
        print("No base models specified. Exiting.")
        return

    # Define base models
    base_models_dict = {
        "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=shape),
        "VGG19": VGG19(weights='imagenet', include_top=False, input_shape=shape),
        "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=shape),
        "ResNet101": ResNet101(weights='imagenet', include_top=False, input_shape=shape),
        "InceptionV3": InceptionV3(weights='imagenet', include_top=False, input_shape=shape),
        "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=shape),
        "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=shape),
        "Xception": Xception(weights='imagenet', include_top=False, input_shape=shape),
        "InceptionResNetV2": InceptionResNetV2(weights='imagenet', include_top=False, input_shape=shape),
        "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=shape)
    }

    for model_name in base_model_names:
        print(f'Training {model_name}...')
        base_model = base_models_dict.get(model_name)
        if base_model is None:
            print(f"Model {model_name} not supported.")
            continue
        create_and_train_model(base_model, model_name, shape, X_train, X_val, num_classes, labels, log_dir, model_dir,
                               epochs, optimizer, learning_rate, step_gamma, alpha, batch_size, patience)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models using transfer learning")
    parser.add_argument('--base_models', type=str, nargs='+', default=[],help='List of base models to use for training. Leave empty to skip model training.')
    parser.add_argument('--shape', type=int, nargs=3, metavar=('HEIGHT', 'WIDTH', 'CHANNELS'), default=(224, 224, 3), help='Input shape of the images (height width channels)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the image data')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use (adam or sgd)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--step_gamma', type=float, default=0.96, help='Gamma value for step learning rate schedule')
    parser.add_argument('--alpha', type=float, default=0.9, help='Alpha for the optimizer (used for SGD)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    args = parser.parse_args()
    main(args.base_models, tuple(args.shape), args.data_path, args.log_dir, args.model_dir,
         args.epochs, args.optimizer, args.learning_rate, args.step_gamma, args.alpha, args.batch_size, args.patience)
