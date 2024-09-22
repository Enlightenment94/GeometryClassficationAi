import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_height, img_width = 200, 200
image_dir = 'random_figure'

def load_images_from_directory(directory, img_height, img_width):
    images = []
    filenames = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            filenames.append(filename)
    
    return np.array(images), filenames

images, filenames = load_images_from_directory(image_dir, img_height, img_width)
print(f'Liczba wczytanych obrazów: {len(images)}')

model = tf.keras.models.load_model('output/model-mlearn3.h5')

predictions = model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)

class_labels = ['circle', 'ellipse', 'line', 'rectangle', 'triangle']  

def plot_predictions(images, filenames, predicted_classes, class_labels, start_index=0):
    plt.figure(figsize=(15, 15))

    num_images_to_display = len(images) 

    x = int(len(images) / 5) + 1
    y = 5
    for i in range(num_images_to_display):
        plt.subplot(x, y, i + 1)
        plt.imshow(images[start_index + i])
        plt.axis('off')
        
        predicted_label = predicted_classes[start_index + i]
        title = class_labels[predicted_label]
        plt.text(0.5, -0.1, title, fontsize=12, ha='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

plot_predictions(images, filenames, predicted_classes, class_labels)
