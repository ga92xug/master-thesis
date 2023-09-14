import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np

def circle_mask(size):
    """
    Generates a circular mask for the given size.

    Parameters:
    - size: tuple (height, width) specifying the size of the mask.

    Returns:
    - A binary mask where the circle region is 1 and the rest is 0.
    """
    height, width = size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    circle_radius = min(width, height) // 2
    left_up_point = (width - circle_radius * 2) // 2
    right_down_point = (width + circle_radius * 2) // 2
    draw.ellipse([left_up_point, left_up_point, right_down_point, right_down_point], fill=1)
    mask = torch.tensor(np.array(mask), dtype=torch.float32)
    return mask

def circle_crop(image_tensor, mask):
    """
    Applies a circular crop to the given image tensor.

    Parameters:
    - image_tensor: the input image tensor of shape [C, H, W].
    - mask: the circular mask tensor of shape [H, W].

    Returns:
    - Image tensor with circular region preserved and rest set to zero.
    """
    expanded_mask = mask.unsqueeze(0).expand_as(image_tensor)
    return image_tensor * expanded_mask

"""
# Load your image
image_path = "path_to_your_image.jpg"
image = Image.open(image_path)

# Convert image to tensor
to_tensor = T.ToTensor()
image_tensor = to_tensor(image)

# Generate circle mask
mask = circle_mask(image_tensor.shape[1:])

# Apply circular crop
cropped_tensor = circle_crop(image_tensor, mask)

# Rotate the image
angle = 45  # For demonstration, rotate by 45 degrees.
rotate_transform = T.Compose([
    T.ToPILImage(),
    T.functional.rotate(angle, resample=Image.BILINEAR),
    T.ToTensor()
])
rotated_tensor = rotate_transform(cropped_tensor)

# Convert tensor back to image and display
to_image = T.ToPILImage()
rotated_image = to_image(rotated_tensor)
rotated_image.show()
"""