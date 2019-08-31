import glob
import random
import shutil

# define number of image to selecet, source and destination folder
src_path = "/home/trojrobert/SRP/dataset/asphalt_data"
dest_path = "/home/trojrobert/SRP/selected_dataset/validation/asphalt"
num_seleceted_image = "" "#50

# select images from source folder and move them to destination folder
image_list = glob.glob(src_path + "/*.png")
selected_images = random.sample(image_list, k=num_seleceted_image)
for file_name in selected_images:
    move_selected = shutil.move(file_name, dest_path)
