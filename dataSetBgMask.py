import os
import shutil
from PIL import Image, ImageDraw
import numpy as np

width, height = 200, 200

train_dir = 'train_dataset_bg_m'
test_dir = 'test_dataset_bg_m'
masks_dir = 'masks_dataset_bg_m'
masks_test_dir = 'masks_test_dataset_bg_m'

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

if os.path.exists(masks_dir):
    shutil.rmtree(masks_dir)

if os.path.exists(masks_test_dir):
    shutil.rmtree(masks_test_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)
os.makedirs(masks_test_dir, exist_ok=True)

shapes = ['rectangle', 'circle', 'ellipse', 'triangle', 'line']

for shape in shapes:
    os.makedirs(os.path.join(train_dir, shape), exist_ok=True)
    os.makedirs(os.path.join(test_dir, shape), exist_ok=True)
    os.makedirs(os.path.join(masks_dir, shape), exist_ok=True)
    os.makedirs(os.path.join(masks_test_dir, shape), exist_ok=True)

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

def generate_mask(shape, size):
    """Generate a binary mask for the given shape and size."""
    mask_image = Image.new('L', (width, height), 0)  
    mask_draw = ImageDraw.Draw(mask_image)
    
    if shape == 'rectangle':
        top_left = (width // 2 - size // 2, height // 2 - size // 2)
        bottom_right = (width // 2 + size // 2, height // 2 + size // 2)
        mask_draw.rectangle([top_left, bottom_right], fill=255)
    elif shape == 'circle':
        bounding_box = [width // 2 - size // 2, height // 2 - size // 2,
                        width // 2 + size // 2, height // 2 + size // 2]
        mask_draw.ellipse(bounding_box, fill=255)
    elif shape == 'ellipse':
        size_x = np.random.randint(30, 100)
        size_y = np.random.randint(30, 100)
        bounding_box = [width // 2 - size_x // 2, height // 2 - size_y // 2,
                        width // 2 + size_x // 2, height // 2 + size_y // 2]
        mask_draw.ellipse(bounding_box, fill=255)
    elif shape == 'triangle':
        points = [
            (width // 2, height // 2 - size // 2),
            (width // 2 + size // 2, height // 2 + size // 2),
            (width // 2 - size // 2, height // 2 + size // 2)
        ]
        mask_draw.polygon(points, fill=255)
    elif shape == 'line':
        start = (width // 2 - size // 2, height // 2)
        end = (width // 2 + size // 2, height // 2)
        mask_draw.line([start, end], fill=255, width=5)
    
    return mask_image

def generate_image(file_path, mask_path, shape, color, size, rotation_angle):
    """Generate an image with a given shape, color, size, rotation angle, and random position."""
    background_color = (255, 255, 255)  
    #image = Image.new('RGB', (width, height), background_color)
    image = generate_random_background(width, height)

    shape_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    mask_image = Image.new('L', (width, height), 0)

    shape_draw = ImageDraw.Draw(shape_image)
    mask_draw = ImageDraw.Draw(mask_image)

    x_pos = np.random.randint(0, width - size)
    y_pos = np.random.randint(0, height - size)

    if shape == 'rectangle':
        top_left = (x_pos, y_pos)
        bottom_right = (x_pos + size, y_pos + size)
        shape_draw.rectangle([top_left, bottom_right], fill=color)
        mask_draw.rectangle([top_left, bottom_right], fill=255)
    elif shape == 'circle':
        bounding_box = [x_pos, y_pos, x_pos + size, y_pos + size]
        shape_draw.ellipse(bounding_box, fill=color)
        mask_draw.ellipse(bounding_box, fill=255)
    elif shape == 'ellipse':
        size_x = np.random.randint(30, 100)
        size_y = np.random.randint(30, 100)
        bounding_box = [x_pos, y_pos, x_pos + size_x, y_pos + size_y]
        shape_draw.ellipse(bounding_box, fill=color)
        mask_draw.ellipse(bounding_box, fill=255)
    elif shape == 'triangle':
        points = [
            (x_pos + size // 2, y_pos),  
            (x_pos + size, y_pos + size),  
            (x_pos, y_pos + size)  
        ]
        shape_draw.polygon(points, fill=color)
        mask_draw.polygon(points, fill=255)
    elif shape == 'line':
        start = (x_pos, y_pos + size // 2)
        end = (x_pos + size, y_pos + size // 2)
        shape_draw.line([start, end], fill=color, width=5)
        mask_draw.line([start, end], fill=255, width=5)

    shape_image = shape_image.rotate(rotation_angle, expand=False, resample=Image.BICUBIC)
    mask_image = mask_image.rotate(rotation_angle, expand=False, resample=Image.BICUBIC)

    rotated_shape_box = shape_image.getbbox() 
    if rotated_shape_box:
        shift_x = max(0, rotated_shape_box[2] - width)
        shift_y = max(0, rotated_shape_box[3] - height)

        shifted_shape_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        shifted_mask_image = Image.new('L', (width, height), 0)
        shifted_shape_image.paste(shape_image, (-shift_x, -shift_y), shape_image)
        shifted_mask_image.paste(mask_image, (-shift_x, -shift_y), mask_image)
    else:
        shifted_shape_image = shape_image
        shifted_mask_image = mask_image

    image.paste(shifted_shape_image, (0, 0), shifted_shape_image)

    image.save(file_path)
    shifted_mask_image.save(mask_path)


for i in range(200):
    shape = np.random.choice(shapes)
    color = tuple(np.random.randint(0, 256, size=3))  
    size = np.random.randint(30, 100)  
    rotation_angle = np.random.randint(0, 360) 
    file_path = os.path.join(train_dir, shape, f'image_{i}.png')
    mask_path_temp = os.path.join(masks_dir, shape, f'mask_{i}.png')  
    generate_image(file_path, mask_path_temp, shape, color, size, rotation_angle)

for i in range(20):
    shape = np.random.choice(shapes)
    color = tuple(np.random.randint(0, 256, size=3))  
    size = np.random.randint(30, 100) 
    rotation_angle = np.random.randint(0, 360) 
    file_path = os.path.join(test_dir, shape, f'image_{i}.png')
    masks_test_dir_temp = os.path.join(masks_test_dir, shape, f'mask_{i}.png')  
    generate_image(file_path, masks_test_dir_temp, shape, color, size, rotation_angle)
