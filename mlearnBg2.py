import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_height, img_width = 200, 200
batch_size = 16
class_labels = ['circle', 'ellipse', 'line', 'rectangle', 'triangle']

def image_mask_generator(image_datagen, mask_datagen, image_dir, mask_dir, batch_size, img_size):
    """Custom generator for loading images and corresponding masks."""
    image_gen = image_datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical', 
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
        mask_batch = np.where(mask_batch.squeeze() > 0, 1, 0) 
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
    label_input = tf.keras.layers.Input(shape=(5,), name='label_input') 
    
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
        
    mask_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask_output')(c5)  
    
    flat = tf.keras.layers.Flatten()(c5)
    class_output = tf.keras.layers.Dense(5, activation='softmax', name='class_output')(flat)

    model = tf.keras.models.Model(inputs=[inputs, label_input], outputs=[mask_output, class_output])
    return model

input_shape = (img_height, img_width, 3)
model = unet_model(input_shape)
#model.compile(optimizer='adam', loss={'mask_output': 'binary_crossentropy', 'class_output': 'categorical_crossentropy'},metrics=['accuracy'])
model.compile(
    optimizer='adam', 
    loss={
        'mask_output': 'binary_crossentropy', 
        'class_output': 'categorical_crossentropy'
    },
    metrics={
        'mask_output': 'accuracy', 
        'class_output': 'accuracy'
    }
)
model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=2, 
    epochs=1,
    validation_data=validation_generator,
    validation_steps=3
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=5)
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
plt.plot(history_dict['mask_output_accuracy'], color='blue', label='Training Mask Accuracy')
plt.plot(history_dict['val_mask_output_accuracy'], color='red', label='Validation Mask Accuracy')
plt.title('Mask Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
#plt.show()


def visualize_predictions(model, generator, num_images=5, class_labels=None):
    for _ in range(num_images):
        image_batch, mask_batch = next(generator)
        pred_mask, class_output = model.predict(image_batch)

        for i in range(len(image_batch)):
            predicted_class_index = np.argmax(class_output[i])

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(image_batch[i][0])
            plt.title(f'Image\nPredicted Class: {class_labels[predicted_class_index]}')

            plt.subplot(1, 3, 2)
            plt.imshow(mask_batch[i], cmap='gray')
            plt.title('True Mask')

            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask[i], cmap='gray')
            plt.title('Predicted Mask')

            plt.show()
            break

visualize_predictions(model, test_generator, num_images=5, class_labels=class_labels)

