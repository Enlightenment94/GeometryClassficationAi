import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
img_height, img_width = 200, 200
batch_size = 16
class_labels = ['circle', 'ellipse', 'line', 'rectangle', 'triangle']

def image_mask_generator(image_datagen, mask_datagen, image_dir, mask_dir, batch_size, img_size):
    """Custom generator for loading images and corresponding masks."""
    image_gen = image_datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        seed=42
    )
    
    mask_gen = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=42
    )
    
    while True:
        image_batch = next(image_gen)
        mask_batch = next(mask_gen)

        mask_batch = np.where(mask_batch > 0, 1, 0)
        
        yield image_batch, mask_batch

# Load the saved model
model = tf.keras.models.load_model('output/model-mlearnBg.h5')

# Create a generator for test images
test_image_datagen = ImageDataGenerator(rescale=1./255)
test_mask_datagen = ImageDataGenerator()

test_generator = image_mask_generator(
    test_image_datagen,
    test_mask_datagen,
    image_dir='test_dataset_bg_m',
    mask_dir='masks_test_dataset_bg_m',
    batch_size=batch_size,
    img_size=(img_height, img_width)
)

def visualize_predictions(model, generator, num_images=5):
    for i in range(num_images):
        image, mask = next(generator)
        pred_mask = model.predict(image)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image[0])
        plt.title('Image')

        plt.subplot(1, 3, 2)
        plt.imshow(mask[0, :, :, 0], cmap='gray')
        plt.title('True Mask')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask[0, :, :, 0], cmap='gray')
        plt.title('Predicted Mask')

        plt.suptitle(f'Predicted Class: {class_labels[int(np.round(pred_mask[0, :, :, 0].mean()))]}')
        plt.show()

# Visualize predictions on the test data
visualize_predictions(model, test_generator, num_images=5)

# Print class labels
print("Class labels:", class_labels)
