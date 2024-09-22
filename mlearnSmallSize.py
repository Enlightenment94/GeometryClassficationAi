import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

img_height, img_width = 28, 28
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    'train_dataset',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'train_dataset',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'test_dataset',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

def build_advanced_dense_model_128(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),  
        tf.keras.layers.Dropout(0.2),                                
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_advanced_dense_model_256_128(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), 
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_advanced_dense_model_512_256(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),  
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = build_advanced_dense_model_512_256((img_height, img_width, 3), len(train_generator.class_indices))
#model = build_advanced_dense_model_256_128((img_height, img_width, 3), len(train_generator.class_indices))
#model = build_advanced_dense_model_128((img_height, img_width, 3), len(train_generator.class_indices))

plot_model(model, to_file='mlearnSmallSize.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

model.save('output/mlearnSmallSize.h5')

history_dict = history.history

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(len(history_dict['loss'])), history_dict['loss'], color='blue', alpha=0.6, label='Trening')
plt.bar(range(len(history_dict['val_loss'])), history_dict['val_loss'], color='red', alpha=0.6, label='Walidacja')
plt.title('Strata')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(range(len(history_dict['accuracy'])), history_dict['accuracy'], color='blue', alpha=0.6, label='Trening')
plt.bar(range(len(history_dict['val_accuracy'])), history_dict['val_accuracy'], color='red', alpha=0.6, label='Walidacja')
plt.title('Dokładność')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.tight_layout()
plt.show()

