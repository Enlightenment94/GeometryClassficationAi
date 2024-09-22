import os
from PIL import Image, ImageDraw
import numpy as np

width, height = 200, 200

train_dir = 'train_dataset_bg'
test_dir = 'test_dataset_bg'

shapes = ['rectangle', 'circle', 'ellipse', 'triangle', 'line']

for shape in shapes:
    os.makedirs(os.path.join(train_dir, shape), exist_ok=True)
    os.makedirs(os.path.join(test_dir, shape), exist_ok=True)

def generate_random_background(image_width, image_height):
    """Generates a random patchy background with colorful spots."""
    background = Image.new('RGB', (image_width, image_height))
    draw = ImageDraw.Draw(background)

    for _ in range(500):
        color = tuple(np.random.randint(0, 256, size=3))  
        patch_size = np.random.randint(10, 30) 
        x0 = np.random.randint(0, image_width)
        y0 = np.random.randint(0, image_height)
        x1 = x0 + patch_size
        y1 = y0 + patch_size
        draw.rectangle([x0, y0, x1, y1], fill=color)
    
    return background

def generate_image(file_path, shape, color, size, rotation_angle):
    """Generate an image with a given shape, color, size, and rotation angle."""
    image = generate_random_background(width, height)
    
    shape_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    shape_draw = ImageDraw.Draw(shape_image)
    
    if shape == 'rectangle':
        top_left = (width//2 - size//2, height//2 - size//2)
        bottom_right = (width//2 + size//2, height//2 + size//2)
        shape_draw.rectangle([top_left, bottom_right], fill=color)
    
    elif shape == 'circle':
        bounding_box = [width//2 - size//2, height//2 - size//2, width//2 + size//2, height//2 + size//2]
        shape_draw.ellipse(bounding_box, fill=color)
    
    elif shape == 'ellipse':
        size_x = np.random.randint(30, 100)
        size_y = np.random.randint(30, 100)
        bounding_box = [width//2 - size_x//2, height//2 - size_y//2, width//2 + size_x//2, height//2 + size_y//2]
        shape_draw.ellipse(bounding_box, fill=color)
    
    elif shape == 'triangle':
        points = [
            (width//2, height//2 - size//2),
            (width//2 + size//2, height//2 + size//2),
            (width//2 - size//2, height//2 + size//2)
        ]
        shape_draw.polygon(points, fill=color)
    
    elif shape == 'line':
        start = (width//2 - size//2, height//2)
        end = (width//2 + size//2, height//2)
        shape_draw.line([start, end], fill=color, width=5)
    
    shape_image = shape_image.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
    image.paste(shape_image, (0, 0), shape_image)
    image.save(file_path)

for i in range(10000):
    shape = np.random.choice(shapes)
    color = tuple(np.random.randint(0, 256, size=3))  
    size = np.random.randint(30, 100)  
    rotation_angle = np.random.randint(0, 360) 
    file_path = os.path.join(train_dir, shape, f'image_{i}.png')
    generate_image(file_path, shape, color, size, rotation_angle)

for i in range(1000):
    shape = np.random.choice(shapes)
    color = tuple(np.random.randint(0, 256, size=3))  
    size = np.random.randint(30, 100) 
    rotation_angle = np.random.randint(0, 360) 
    file_path = os.path.join(test_dir, shape, f'image_{i}.png')
    generate_image(file_path, shape, color, size, rotation_angle)
