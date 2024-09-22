import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

img_height, img_width = 200, 200
batch_size = 16

def image_mask_generator(image_datagen, mask_datagen, image_dir, mask_dir, batch_size, img_size):
    """Custom generator for loading images and corresponding masks."""
    image_gen = image_datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='None',  
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


train_image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_mask_datagen = ImageDataGenerator(validation_split=0.2)

train_generator = image_mask_generator(
    train_image_datagen,
    train_mask_datagen,
    image_dir='train_dataset_bg_m',
    mask_dir='masks_dataset_bg_m',
    batch_size=batch_size,
    img_size=(img_height, img_width)
)

validation_generator = image_mask_generator(
    train_image_datagen,
    train_mask_datagen,
    image_dir='train_dataset_bg_m',
    mask_dir='masks_dataset_bg_m',
    batch_size=batch_size,
    img_size=(img_height, img_width)
)

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

def unet_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    
    u1 = tf.keras.layers.UpSampling2D((2, 2))(c3)
    concat1 = tf.keras.layers.Concatenate()([u1, c2])
    c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    
    u2 = tf.keras.layers.UpSampling2D((2, 2))(c4)
    concat2 = tf.keras.layers.Concatenate()([u2, c1])
    c5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)  
    
    model = tf.keras.models.Model(inputs, outputs)
    return model

input_shape = (img_height, img_width, 3)
model = unet_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=100, 
    epochs=2,
    validation_data=validation_generator,
    validation_steps=50
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=20)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

model.save('output/model-mlearnBg.h5')

history_dict = history.history

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], color='blue', label='Training Loss')
plt.plot(history_dict['val_loss'], color='red', label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_dict['accuracy'], color='blue', label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], color='red', label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

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

        plt.show()

visualize_predictions(model, test_generator, num_images=5)
