import os

images_directory_path = "C:/Users/Ayaz/Desktop/Python-pdf/fish_detection/yolov7-main/inference/images"

image_names = os.listdir(images_directory_path)

labels_directory_path = "C:/Users/Ayaz/Desktop/Python-pdf/mirror_and_rotate/labels"
label_names = os.listdir(labels_directory_path)

output_folder_path = "C:/Users/Ayaz/Desktop/Python-pdf/mirror_and_rotate/output"
images_and_labels = tuple(zip(image_names, label_names))

for image_name in image_names:

    image_path = os.path.join(images_directory_path, image_name)
    print(image_path)
