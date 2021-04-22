import os


image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(base_path):
    for (root_dir, dir_names, file_names) in os.walk(base_path):
        for file_name in file_names:
            file_extension = file_name[file_name.rfind("."):].lower()
            if file_extension.endswith(image_types):
                image_path = os.path.join(root_dir, file_name)
                yield image_path
