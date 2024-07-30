import numpy as np
from lime import lime_image
from lime.lime_image import LimeImageExplainer, SegmentationAlgorithm

def generate_lime_mask(img_array, model, num_samples, num_features):
    explainer = LimeImageExplainer()
    explanation_instance = explainer.explain_instance(
        img_array, model.predict, top_labels=1, hide_color=0, num_samples=num_samples, num_features=num_features
    )
    temp, mask = explanation_instance.get_image_and_mask(
        explanation_instance.top_labels[0], positive_only=True, num_features=num_features, hide_rest=True
    )
        
    # Ensure img_array is 3-dimensional (height, width, channels)
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    
    lime_mask = np.zeros_like(img_array)
    lime_mask[mask == 1] = img_array[mask == 1]
    
    masked_image = np.copy(img_array)
    masked_image[mask == 0] = 255  # convert black to white background

    return masked_image, explanation_instance

def generate_splime_mask(img_array, model, num_features, num_samples, segmentation_alg, kernel_size, max_dist, ratio):
    explainer = LimeImageExplainer()
    if segmentation_alg == 'quickshift':
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    else:
        segmentation_fn = SegmentationAlgorithm('slic', n_segments=kernel_size, compactness=max_dist, sigma=ratio)
    
    explanation = explainer.explain_instance(
        img_array, model.predict, top_labels=1, hide_color=0,
        num_samples=num_samples, num_features=num_features, segmentation_fn=segmentation_fn
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=num_features, hide_rest=True
    )
    
    # Ensure the mask is in the correct format
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
    if mask.shape[-1] == 1:
        mask = np.repeat(mask, 3, axis=-1)

    # Create a black image
    black_image = np.zeros_like(img_array)
    # Apply the mask to the image
    mask_image = np.copy(black_image)
    mask_image[mask == 1] = img_array[mask == 1]

    return mask_image