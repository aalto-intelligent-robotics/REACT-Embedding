import os
from PIL import Image

# Get the list of all files in the current directory
files = os.listdir('.')

# Loop through all files
for file_name in files:
    # Check if the file is a PNG image
    if file_name.lower().endswith('.png'):
        try:
            # Open the image
            with Image.open(file_name) as img:
                # Flip the image horizontally
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Create the new filename
                new_file_name = f"{os.path.splitext(file_name)[0]}_flipped.png"
                
                # Save the flipped image
                flipped_img.save(new_file_name)
                print(f"Saved flipped image as {new_file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
