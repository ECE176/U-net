import os
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import matplotlib.gridspec as gridspec


def load_coco_annotations(data_dir):
    annotation_file = os.path.join(data_dir, '_annotations.coco.json')
    with open(annotation_file, 'r') as file:
        coco_annotations = json.load(file)
    return coco_annotations

def visualize_annotations(image, annotations, ax):
    for ann in annotations['annotations']:
        # Assuming the image has been resized to 640x640
        # Draw segmentation
        for seg in ann['segmentation']:
            poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            polygon = patches.Polygon(poly, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)
    ax.imshow(image)
    ax.axis('off')


def visualize_annotations_grid(data_dir):
    coco_annotations = load_coco_annotations(data_dir)
    coco = COCO(os.path.join(data_dir, '_annotations.coco.json'))

    # Setup the matplotlib grid for 2x2 images
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 2)

    # Load and display the first 4 images

    for idx in range(4):  # Loop through the first four images
        img_id = coco.getImgIds()[idx]  # Get the image ID
        img_info = coco.loadImgs(img_id)[0]  # Load the image information
        image_path = os.path.join(data_dir, img_info['file_name'])  # Construct the image file path
        image = cv2.imread(image_path)  # Load the image
        
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))  # Load annotations for the image
        
        ax = plt.subplot(gs[idx])  # Create a subplot in the grid
        ax.axis('off')  # Turn off the axis
        ax.set_aspect('equal')  # Set the aspect ratio to equal

        visualize_annotations(image, {'annotations': annotations}, ax)  # Visualize annotations on the image

    plt.show()
