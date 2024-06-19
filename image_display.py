# # image_display.py
# from PIL import Image
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# def display_image(inputs, save_path="output_image.png"):
#     transform = transforms.ToPILImage()
#     image = transform(inputs[-1])
#     image.save(save_path)
#     print(f"Image saved to {save_path}")



from PIL import Image
import torchvision.transforms as transforms

def display_image(inputs, save_path="output_image.png", rows=4, cols=8):
    transform = transforms.ToPILImage()
    images = [transform(img) for img in inputs[:rows * cols]]
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    combined_image = Image.new("RGB", (width * cols, height * rows))
    for i, img in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        combined_image.paste(img, (x, y))
    combined_image.save(save_path)