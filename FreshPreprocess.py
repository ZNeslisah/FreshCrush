import os
from rembg import remove
from PIL import Image

# Directories for your data
current_dir = os.path.dirname(os.path.abspath(__file__))
# train_dir = os.path.join(current_dir, "data2", "train", "banana")
# train_dir = os.path.join(current_dir, "data2", "train", "apple")
# train_dir = os.path.join(current_dir, "data2", "train", "orange")

# val_dir = os.path.join(current_dir, "data2", "val", "banana")
# val_dir = os.path.join(current_dir, "data2", "val", "apple")
# val_dir = os.path.join(current_dir, "data2", "val", "orange")

# test_dir = os.path.join(current_dir, "data2", "test", "apple")

# Output directories for processed images
# output_train_dir = os.path.join(current_dir, "data2_processed", "train", "banana")
# output_train_dir = os.path.join(current_dir, "data2_processed", "train", "apple")
# output_train_dir = os.path.join(current_dir, "data2_processed", "train", "orange")
# output_val_dir = os.path.join(current_dir, "data2_processed", "val", "banana")
# output_val_dir = os.path.join(current_dir, "data2_processed", "val", "apple")
# output_val_dir = os.path.join(current_dir, "data2_processed", "val", "orange")
# output_test_dir = os.path.join(current_dir, "data2_processed", "test", "apple")

# Create output directories if they don't exist
# os.makedirs(output_train_dir, exist_ok=True)
# os.makedirs(output_val_dir, exist_ok=True)
# os.makedirs(output_test_dir, exist_ok=True)

# image= os.path.join(current_dir, "huge data", "Train_Set_Folder", "cucumber")
# image_change = os.path.join(current_dir, "huge data processed", "Train_Set_Folder", "cucumber")
image= os.path.join(current_dir, "fruitCrash", "fruitCrash_dataset","level2_original gercek")
image_change = os.path.join(current_dir, "fruitCrash", "fruitCrash_dataset", "level2_son2")

os.makedirs(image_change, exist_ok=True)


def process_images(input_dir, output_dir):
    """Remove background from all images in the input directory and save to output directory."""
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                with open(input_path, "rb") as inp_file:
                    input_data = inp_file.read()

                output_data = remove(input_data)  # Remove background

                with open(output_path, "wb") as out_file:
                    out_file.write(output_data)
                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

# Process each directory
# process_images(train_dir, output_train_dir)
# process_images(val_dir, output_val_dir)
# process_images(test_dir, output_test_dir)

process_images(image, image_change)

print("Background removal complete for all datasets.")
