from PIL import Image
import os

image_width = 240
image_height = 240

save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)
blank_image = Image.new('L', (image_width, image_height), color=0)

filename = "normal_mask.PNG"
filepath = os.path.join(save_dir, filename)
blank_image.save(filepath)

print(f"Blank image {filename} saved.")