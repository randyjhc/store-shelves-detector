# Import necessary libraries
import cv2, os, random, shutil, click, kagglehub
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt


def subsample_data(source_path, target_path, num_samples):
    """
    Randomly subsample a specified number of images and labels from the source dataset
    and copy them to a target directory.

    Parameters:
        source_path (str): Path to the source dataset.
        target_path (str): Path to store the subsampled data.
        num_samples (tuple): Number of samples for train, validation, and test sets.

    Returns:
        tuple: Lists of image and label file paths.
    """
    # Define directory suffixes for images and labels
    image_path_suffix = ["images/train/", "images/val/", "images/test/"]
    label_path_suffix = ["labels/train/", "labels/val/", "labels/test/"]

    # Lists to store the paths of selected images and labels
    chosen_image_paths, chosen_label_paths = [], []

    # Loop over train, val, and test directories
    for i in range(3):
        # Construct paths for source and target directories
        source_image_dir = source_path + image_path_suffix[i]
        source_label_dir = source_path + label_path_suffix[i]
        target_image_dir = target_path + image_path_suffix[i]
        target_label_dir = target_path + label_path_suffix[i]

        # Create directories for the subset if they don't exist
        os.makedirs(target_image_dir, exist_ok=True)
        os.makedirs(target_label_dir, exist_ok=True)

        # List all file names in the source directory
        file_names = os.listdir(source_image_dir)

        # Randomly select the specified number of samples
        random.seed(88)
        chosen_file_names = random.choices(file_names, k=num_samples[i])

        # Copy selected files to the target directory
        for file_path in chosen_file_names:
            shutil.copy(source_image_dir + file_path, target_image_dir + file_path)
            shutil.copy(
                source_label_dir + file_path[:-3] + "txt",
                target_label_dir + file_path[:-3] + "txt",
            )
            # Save the paths of the chosen files
            chosen_image_paths.append(target_image_dir + file_path)
            chosen_label_paths.append(target_label_dir + file_path[:-3] + "txt")

    # Print the chosen file paths for debugging
    print(chosen_image_paths)
    print(chosen_label_paths)

    return chosen_image_paths, chosen_label_paths


def image_preprocess(image_paths, target_size=(640, 640), pad_color=(0, 0, 0)):
    """
    Preprocess images by resizing and padding them to a target size.

    Parameters:
        image_paths (list): List of paths to input images.
        target_size (tuple): The desired output size (width, height).
        pad_color (tuple): Color to use for padding (BGR).

    Returns:
        tuple: Processed images and scaling parameters for each.
    """
    image_objects, scaling_parameters = [], []
    for image_path in image_paths:
        # Read the image from the file
        image = cv2.imread(image_path)

        # Calculate the scaling factor to fit the target dimensions
        original_h, original_w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / original_w, target_h / original_h)
        scale_w, scale_h = int(original_w * scale), int(original_h * scale)

        # Resize the image based on the scaling factor
        image_resized = cv2.resize(
            image, (scale_w, scale_h), interpolation=cv2.INTER_AREA
        )

        # Create a new blank image with padding
        padded_image = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

        # Calculate padding offsets
        pad_top = (target_h - scale_h) // 2
        pad_left = (target_w - scale_w) // 2

        # Insert the resized image into the padded image
        padded_image[pad_top : pad_top + scale_h, pad_left : pad_left + scale_w] = (
            image_resized
        )

        # Append processed image and scaling data
        image_objects.append(padded_image)
        scaling_parameters.append(
            (
                pad_left / target_w,
                pad_top / target_h,
                scale_w / target_w,
                scale_h / target_h,
            )
        )

        # Ensure the directory exists for saving processed images
        processed_path = image_path.replace("01_datasets", "02_preprocess")
        print(processed_path)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)

        # Save the processed image
        cv2.imwrite(processed_path, padded_image)

    return image_objects, scaling_parameters


def show_images(input_image):
    """
    Display an image using matplotlib.

    Parameters:
        input_image (numpy array): The image to be displayed.
    """
    image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def prepare_labels(label_paths, scaling_parameters):
    """
    Adjust bounding box labels based on image scaling and padding.

    Parameters:
        label_paths (list): Paths to the label files.
        scaling_parameters (list): Scaling and padding information.

    Returns:
        list: Transformed bounding box coordinates for each label file.
    """
    bbox_objects = []
    for label_path, scale in zip(label_paths, scaling_parameters):
        lines = []
        w_pad, h_pad, w_ratio, h_ratio = scale
        with open(label_path, "r") as fp:
            for line in fp:
                # Parse bounding box values
                bbox = line.split()
                bbox = [
                    int(bbox[0]),
                    float(bbox[1]) * w_ratio + w_pad,
                    float(bbox[2]) * h_ratio + h_pad,
                    float(bbox[3]) * w_ratio,
                    float(bbox[4]) * h_ratio,
                ]
                lines.append(bbox)
        # Save processed label data
        processed_path = label_path.replace("01_datasets", "02_preprocess")
        print(processed_path)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        with open(processed_path, "w") as fp:
            for line in lines:
                fp.write(" ".join(map(str, line)) + "\n")

        bbox_objects.append(lines)
        print(len(bbox_objects[-1]))

    return bbox_objects


def draw_one_image_with_bboxes(image, bounding_boxes):
    """
    Draw bounding boxes on an image and display it.

    Parameters:
        image (numpy array): The input image.
        bounding_boxes (list): List of bounding boxes to draw.
    """
    print(f"Number of labels = {len(bounding_boxes)}")
    for bbox in bounding_boxes:
        _, x_center, y_center, width, height = bbox
        scale_y, scale_x = image.shape[:2]

        # Calculate bounding box corners
        x_min = int((x_center - width / 2) * scale_x)
        y_min = int((y_center - height / 2) * scale_y)
        x_max = int((x_center + width / 2) * scale_x)
        y_max = int((y_center + height / 2) * scale_y)

        # Draw the bounding box on the image
        start_point = (x_min, y_min)
        end_point = (x_max, y_max)
        color = (0, 255, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # Convert the image to RGB and show it
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def write_train_yaml(yaml_name, full_dataset=False):
    """
    Generate a YAML configuration file for YOLO training.

    Parameters:
        yaml_name (str): Name of the YAML file to be created.
        full_dataset (bool): Whether to use the full dataset path or preprocessed.
    """
    with open(yaml_name, "w") as fp:
        if full_dataset:
            fp.write(f"path: {path}/SKU110K_fixed/\n")
        else:
            fp.write(f"path: ../02_preprocess/\n")
        fp.write(f"train: images/train\n")
        fp.write(f"val: images/val\n")
        fp.write(f"nc: 1\n")  # Number of classes
        fp.write(f"names: ['object']")  # Class names


def read_image_list(image_list):
    """
    Reads a list of image file paths from a text file and returns them as a list of strings.

    Args:
        image_list (str): The path to a text file containing the list of image file paths.
                          Each line in the file should represent a separate image file path.

    Returns:
        list of str: A list containing the image file paths as strings.
    """
    with open(image_list, "r") as fp:
        image_filenames = [line.rstrip() for line in fp]
    return image_filenames


# The Click library command-line interface setup
@click.command()
@click.option(
    "--samples",
    type=click.Tuple([int, int, int]),
    default=(80, 20, 20),
    help="tuple of (train, val, test) samples"
)
@click.option(
    "--weights",
    type=click.STRING,
    default="yolo11n.pt",
    help="string of pre-trained model name"
)
@click.option(
    "--epochs", type=click.INT, default=1, help="number of training iterations"
)
def train(samples, weights, epochs):
    """
    Train a YOLOv11 model using the specified dataset and parameters.\f

    Parameters:
        samples (tuple): Number of samples for training, validation, and testing.
        weights (str): Path to the pre-trained weights.
        epochs (int): Number of training epochs.
    """
    click.echo(click.style("=== Start training process. ===", fg="magenta"))

    # Download dataset from Kaggle
    path = kagglehub.dataset_download("thedatasith/sku110k-annotations")
    print("Path to dataset files:", path)

    # Subsample datasets and preprocess images and labels
    image_paths, label_paths = subsample_data(
        path + "/SKU110K_fixed/", "./01_datasets/", samples
    )
    image_objects, scaling_parameters = image_preprocess(image_paths)
    for i in range(3):
        show_images(image_objects[i])
        print(scaling_parameters[i])
    bbox_objects = prepare_labels(label_paths, scaling_parameters)

    for i in range(5):
        draw_one_image_with_bboxes(image_objects[i], bbox_objects[i])

    # Write training configuration
    write_train_yaml("train.yaml", full_dataset=False)

    # Load pre-trained model and start training
    model = YOLO(weights)
    train_results = model.train(
        data="train.yaml",
        epochs=epochs,
        imgsz=640,
        device="cpu",
        save_period=1,
    )

    # Evaluate the model
    metrics = model.val()

    click.echo(
        click.style("=== The training completed successfully. ===", fg="magenta")
    )


@click.command()
@click.argument("model", type=click.STRING)
@click.argument("images", type=click.STRING)
@click.option(
    "--save_dir",
    type=click.Tuple([str, str]),
    default=("samples", "outputs"),
    help="string of folder name for prediction output",
)
@click.option(
    "--verbose",
    type=click.BOOL,
    default=False,
    help="set True to show image after prediction",
)
def predict(model, images, save_dir=("samples", "outputs"), verbose=False):
    """
    Predict bounding boxes for an image using a pre-trained YOLO model.\f
    
    Parameters:
        model (str): Path to the trained model weights.
        image (str): Path to the input image.
        save_dir (tuple): Path (./tup[0]/tup[1]/) to save the prediction output.
        verbose (bool): Set True to show image after prediction.
    """
    click.echo(click.style("=== Start predicting process. ===", fg="green"))
    # Load model
    model = YOLO(model)
    # Start prediction
    results = model.predict(
        source=read_image_list(images),
        conf=0.4,
        save=True,
        project=save_dir[0],
        name=save_dir[1],
        line_width=3,
        show_labels=True,
        show_conf=True,
    )
    # Display prediction
    if verbose:
        results[0].show()
    click.echo(click.style("=== Prediction completed successfully. ===", fg="green"))


@click.group()
def cli():
    """Main command-line interface group."""
    pass


# Add the training and prediction commands to the CLI
cli.add_command(train)
cli.add_command(predict)

if __name__ == "__main__":
    cli()
