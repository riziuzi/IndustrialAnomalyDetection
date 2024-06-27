# # image_display.py
# from PIL import Image
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# def display_image(inputs, save_path="output_image.png"):
#     transform = transforms.ToPILImage()
#     image = transform(inputs[-1])
#     image.save(save_path)
#     print(f"Image saved to {save_path}")


import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np

def peek_localization(array_3d_list, image_tensor_list, ind, b, _count):
        # Determine global min and max across all arrays in array_3d_list
        global_vmin = np.min([np.min(array_3d.cpu().numpy().reshape(-1)) for array_3d in array_3d_list])
        global_vmax = np.max([np.max(array_3d.cpu().numpy().reshape(-1)) for array_3d in array_3d_list])
        # count = 0
        count = _count                                                          #omit

        # for array_3d, image_tensor in zip(array_3d_list,image_tensor_list):   #Open
        array_3d = array_3d_list[-1]                                            #Omit
        image_tensor = image_tensor_list[len(array_3d_list)-1]                  #Omit
        num_arrays, length, _ = array_3d.shape
        
        if num_arrays != 3 or length != 225:
            raise ValueError("Input array_3d should have shape (3, 225, 1).")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # Transpose to HWC format
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        axes[0, 0].imshow(image_pil)
        axes[0, 0].set_title("Corresponding Image")
        axes[0, 0].axis('off')
        
        for i in range(num_arrays):
            array_1d = array_3d[i].reshape(-1)
            array_2d = array_1d.cpu().numpy().reshape((15, 15))
            
            greater_than_point_one = array_2d[array_2d > 0.2]
            if greater_than_point_one.size > 0:
                avg_greater_than_point_one = np.mean(greater_than_point_one)
            else:
                avg_greater_than_point_one = 0
            
            row = (i + 1) // 2
            col = (i + 1) % 2
            
            sns.heatmap(array_2d, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False, vmin=global_vmin, vmax=global_vmax, ax=axes[row, col])
            axes[row, col].set_title(f"15x15 Heatmap of Values (Avg > 0.2: {avg_greater_than_point_one:.2f}) - Array {i+1}")
            axes[row, col].set_xlabel("Column")
            axes[row, col].set_ylabel("Row")
            
        plt.tight_layout()
        os.makedirs(f'./OUTPUT_images/etc/batch_{b*ind}/', exist_ok=True)
        # plt.savefig(f"./OUTPUT_images/etc/batch_{b*ind}/combined_output_{b*ind+count}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"./OUTPUT_images/etc/batch_{b*ind}/combined_output_{b*ind+count}.png", dpi=300, bbox_inches='tight')
        count+=1
        plt.close()

def save_images_and_patches(img, patch_ref_map, num_images=8):
    indices = torch.randperm(img.size(0))[:num_images]

    selected_images = img[indices]
    grid_img = vutils.make_grid(selected_images, nrow=8, padding=2, normalize=True)
    plt.figure(figsize=(20, 5))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title('Randomly Selected Query Images')
    plt.savefig('selected_query_images.png')
    plt.close()

    selected_patch_maps = patch_ref_map[indices].cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(20, 5)) 
    for i, ax in enumerate(axes.flat):
        if i < len(selected_patch_maps):
            sns.heatmap(selected_patch_maps[i].reshape(15, 15), ax=ax, cbar=False, cmap='viridis') 
        ax.axis('off')
    fig.suptitle('Corresponding Patch Reference Maps')
    plt.savefig('patch_reference_maps.png')
    plt.close()

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

def normalize_to_uint8(image):
    """
    Normalize an image with arbitrary range to uint8.
    
    Parameters:
    image (numpy.ndarray or torch.Tensor): Input image array or tensor.
    
    Returns:
    numpy.ndarray: Normalized image array.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Convert to NumPy array

    min_val = image.min()
    max_val = image.max()
    normalized_image = 255 * (image - min_val) / (max_val - min_val)
    return normalized_image.astype(np.uint8)

def save_images_and_localisation(imgs, localisations, save_dir="./TEST"):          # imgs -> (32, 3, 240, 240); localisations -> (32, 1, 240, 240)
    """
    Save each image and its corresponding localization in a 1x2 grid.

    Parameters:
    imgs (numpy.ndarray): Array of images of shape (32, 3, 240, 240).
    localisations (numpy.ndarray): Array of localization images of shape (32, 1, 240, 240).
    save_dir (str): Directory to save the images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    num_images = imgs.shape[0]
    for i in range(num_images):
        img = imgs[i]
        loc = localisations[i]

        img = normalize_to_uint8(img.permute(1, 2, 0))
        loc = normalize_to_uint8(loc[0])

        img_pil = Image.fromarray(img)
        loc_pil = Image.fromarray(loc).convert('L')

        grid_width = img_pil.width + loc_pil.width
        grid_height = max(img_pil.height, loc_pil.height)
        grid_img = Image.new('RGB', (grid_width, grid_height))

        grid_img.paste(img_pil, (0, 0))
        grid_img.paste(loc_pil.convert('RGB'), (img_pil.width, 0))
        grid_img.save(os.path.join(save_dir, f'image_localisation_{i + 1}.jpg'))