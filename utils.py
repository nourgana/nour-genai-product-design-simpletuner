from PIL import Image, ImageDraw, ImageFont
import os

def combine_images_horizontally(images, output_path, labels):
    """
    Combine images horizontally and save the output in a specified folder. Labels each image.

    :param output_path: Path to the folder where the combined image will be saved
    :param labels: List of labels for each image
    """
    # Load images
    
    # Get the dimensions of the combined image
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    # Create a new image with the combined width and the maximum height
    combined_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # Initialize font for labeling
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Position to paste the next image
    x_offset = 0
    
    for i, img in enumerate(images):
        combined_image.paste(img, (x_offset, 0))
        
        # Draw label
        draw = ImageDraw.Draw(combined_image)
        text_size = draw.textsize(labels[i], font=font)
        text_x = x_offset + (img.width - text_size[0]) // 2
        text_y = img.height - text_size[1] - 10
        draw.text((text_x, text_y), labels[i], font=font, fill="black")
        
        x_offset += img.width
    
    # Ensure the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Save the combined image
    combined_image.save(output_path)
