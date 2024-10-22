import cv2, os, shutil
import numpy as np
import gradio as gr
from shelf_detection import read_image_list, predict


def load_samples(path="input_list.txt"):
    """
    Function to load sample images from a text file

    Parameters:
        path (str): Path to the source image filename list.

    Returns:
        list: Image objects in numpy array.
    """
    filenames = read_image_list(path)
    image_objects = []
    for image_path in filenames:
        image_objects.append(np.array(cv2.imread(image_path)))
    return image_objects


def predict_samples(selected_images):
    """
    Function to predict output images based on selected input images

    Parameters:
        selected_images (list): A list of numpy array containing image objects.

    Returns:
        list: Prediction results in numpy array.
    """

    # If the prediction folder exists, remove it along with its contents
    folder_path = "./03_prediction/"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # Create necessary directories for saving input images
    os.makedirs(os.path.dirname("./03_prediction/inputs/"), exist_ok=True)
    input_list = "./03_prediction/predict_input.txt"
    output_list = "./03_prediction/predict_output.txt"

    # Open the input and output list files for writing
    with open(input_list, "w") as fp_in, open(output_list, "w") as fp_out:
        for i, sample in enumerate(selected_images):
            filename = f"./03_prediction/inputs/predict_{i}.jpg"
            fp_in.write(filename + "\n")
            cv2.imwrite(filename, sample[0])
            fp_out.write(filename.replace("inputs", "outputs") + "\n")

    # Call the predict function with the model weights, input list, and output folder
    predict("./weights/epoch14.pt", input_list, ("03_prediction", "outputs"))

    # Load the predicted images from the output list and return them
    return load_samples(output_list)


# Function to enable the "Predict" button when images are available
def got_inputs(gallery_in):
    return gr.Button.update(interactive=True)


with gr.Blocks() as demo:

    # Display a title and instructions
    gr.Markdown("# Shelf Object Detection")
    gr.Markdown(
        "Upload your image or use generated images and then click **Predict** to see the output."
    )

    # Button to preload sample images
    btn_gen_img = gr.Button("Preload samples", scale=0, min_width=200)

    # Gallery to display chosen images, interactive for image selection
    gallery_in = gr.Gallery(
        label="Chosen images",
        show_label=False,
        elem_id="gallery_in",
        columns=[5],
        rows=[1],
        object_fit="contain",
        height="auto",
        interactive=True,
        type="numpy",
        # value=preview()
    )

    # Button to trigger prediction
    btn_predict = gr.Button("Predict", scale=0, min_width=200)

    # Gallery to display predicted images
    gallery_out = gr.Gallery(
        label="Predicted images",
        show_label=False,
        elem_id="gallery_out",
        columns=[5],
        rows=[1],
        object_fit="contain",
        height="auto",
        type="numpy",
    )
    # event listeners
    btn_gen_img.click(load_samples, None, gallery_in)
    btn_predict.click(predict_samples, gallery_in, gallery_out)

if __name__ == "__main__":
    demo.launch()
