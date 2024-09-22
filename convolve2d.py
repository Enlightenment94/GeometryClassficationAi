import numpy as np
import matplotlib.pyplot as plt

image = np.array([
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 2],
    [7, 8, 9, 2, 3],
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

def convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    output = np.zeros((output_height, output_width))
    
    for y in range(output_height):
        for x in range(output_width):
            region = image[y:y+kernel_height, x:x+kernel_width]
            output[y, x] = np.sum(region * kernel)
    
    return output

convolved_image = convolve2d(image, kernel)

def plot_images(image1, image2, title1, title2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image1, cmap='gray', interpolation='nearest')
    ax[0].set_title(title1)
    ax[0].axis('off')  # Ukrywa osie

    ax[1].imshow(image2, cmap='gray', interpolation='nearest')
    ax[1].set_title(title2)
    ax[1].axis('off')  # Ukrywa osie

    plt.show()

plot_images(image, convolved_image, "Obraz wejściowy", "Obraz po konwolucji")

print("Obraz wejściowy:")
print(image)

print("Filtr (jądro konwolucji):")
print(kernel)

print("Obraz po konwolucji:")
print(convolved_image)
