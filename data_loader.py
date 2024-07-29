import tensorflow as tf
import os
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm  # For progress display
import sys
import uuid  # Import uuid for unique filename generation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Data Loader with Augmentation and Splits')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--dim', type=int, default=224, help='Required image dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--target_folder', type=str, required=True, help='Folder to store the train, test, and val splits')
    parser.add_argument('--augment_data', action='store_true', help='Apply data augmentation')
    return parser.parse_args()

def create_datagens():
    return [
        ImageDataGenerator(rescale=1./255),
        ImageDataGenerator(rotation_range=20),
        ImageDataGenerator(width_shift_range=0.2),
        ImageDataGenerator(height_shift_range=0.2),
        ImageDataGenerator(horizontal_flip=True)
    ]

def process_image(file_path, image_size):
    file_path = file_path.numpy().decode('utf-8')
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def save_image(image, file_path):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.image.encode_jpeg(image)
    tf.io.write_file(file_path, image)

def load_data(path, image_size, batch_size):
    all_images = []
    labels = []

    for subdir, _, files in os.walk(path):
        label = os.path.basename(subdir)
        for fname in files:
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(subdir, fname))
                labels.append(label)
    
    unique_labels = set(labels)
    print(f"Found {len(all_images)} images in {path}\n")
    print(f"Labels found ({len(unique_labels)}): {unique_labels}\n")
    
    if len(all_images) == 0:
        raise ValueError(f"No images found in the specified path: {path}")

    # Stratified splitting the dataset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(sss.split(all_images, labels))
    
    train_files = [all_images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_files = [all_images[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_indices, test_indices = next(sss_val.split(test_files, test_labels))
    
    val_files = [test_files[i] for i in val_indices]
    val_labels = [test_labels[i] for i in val_indices]
    test_files = [test_files[i] for i in test_indices]
    test_labels = [test_labels[i] for i in test_indices]

    print(f"Data split into {len(train_files)} train, {len(val_files)} validation, and {len(test_files)} test images.\n")

    def tf_load_and_augment_image(file_path, label):
        image = tf.py_function(func=lambda x: process_image(x, image_size), inp=[file_path], Tout=tf.float32)
        image.set_shape([image_size, image_size, 3])
        return image, label

    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))

    train_dataset = train_dataset.map(lambda x, y: tf_load_and_augment_image(x, y))
    val_dataset = val_dataset.map(lambda x, y: tf_load_and_augment_image(x, y))
    test_dataset = test_dataset.map(lambda x, y: tf_load_and_augment_image(x, y))

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def save_datasets_to_folders(dataset, folder_path, datagens=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    count = 0
    for batch_images, batch_labels in tqdm(dataset, desc=f"Saving to {folder_path}"):
        for i in range(batch_images.shape[0]):
            image = batch_images[i].numpy()
            label = batch_labels[i].numpy().decode('utf-8')
            label_folder = os.path.join(folder_path, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            # Save the original image
            file_path = os.path.join(label_folder, f"{uuid.uuid4().hex}.jpg")
            save_image(image, file_path)
            count += 1

            # Apply augmentations if datagens are provided
            if datagens:
                for datagen in datagens:
                    aug_image = datagen.random_transform(image)
                    file_path = os.path.join(label_folder, f"{uuid.uuid4().hex}.jpg")
                    save_image(aug_image, file_path)
                    count += 1
    
    print(f"Saved {count} images to {folder_path}\n")
    return count

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.target_folder):
        os.makedirs(args.target_folder)
        
    train_folder = os.path.join(args.target_folder, 'train')
    val_folder = os.path.join(args.target_folder, 'val')
    test_folder = os.path.join(args.target_folder, 'test')
    
    datagens = create_datagens() if args.augment_data else None

    train_dataset, val_dataset, test_dataset = load_data(
        args.path,
        args.dim,
        args.batch_size
    )
    
    # Save datasets to respective folders and count images
    train_count = save_datasets_to_folders(train_dataset, train_folder, datagens)
    val_count = save_datasets_to_folders(val_dataset, val_folder)
    test_count = save_datasets_to_folders(test_dataset, test_folder)
    
    print(f"Train dataset saved to: {train_folder}\n")
    print(f"Validation dataset saved to: {val_folder}\n")
    print(f"Test dataset saved to: {test_folder}\n")
    
    print('-'*20)

    print(f"Number of images in training set: {train_count}\n")
    print(f"Number of images in validation set: {val_count}\n")
    print(f"Number of images in test set: {test_count}\n")

if __name__ == "__main__":
    # Redirect stdout and stderr to avoid encoding issues
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
    main()
