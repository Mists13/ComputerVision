import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np

path_images = './images/'
path_roi = './roi/'

def read_images():
    
    images = []

    for file in glob.glob(path_images + '*.png'):
        img = cv2.imread(file)
        hsv_target = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        images.append(hsv_target)
    return images


target_images = read_images()
# Load the array of ROI images and loop over each one
roi_images = ['roi1.png', 'roi2.png', 'roi3.png', 'roi4.png', 'roi5.png', 'roi6.png']
# roi_images = ['roi6.png','roi2.png']

for roi_image in roi_images:
    for target in target_images:

        # Load the ROI image and convert to HSV color space
        roi_img = cv2.imread(path_roi + roi_image)
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # Set the number of histogram bins to 32
        hist_size = [32, 32]

        # Calculate  and normalize the histogram of the ROI image
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, hist_size, [0, 180, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Calculate the back projection of the target image
        back_proj = cv2.calcBackProject([target], [0, 1], roi_hist, [0, 180, 0, 256], 1)

        # Apply a circular disc filter to the back projection
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(back_proj, -1, disc, back_proj)

        # Threshold the back projection to create a binary image
        _, thresh = cv2.threshold(back_proj, 127, 255, 0)
        thresh = cv2.merge((thresh, thresh, thresh))

        # Apply the binary image as a mask to the original target image
        res = cv2.bitwise_and(target, thresh)

        # # Calculate the similarity score as the sum of the values in the binary image
        # similarity_score = np.sum(thresh[:,:,0]) / 255

        # Convert images from BGR to RGB
        target_img = cv2.cvtColor(target, cv2.COLOR_HSV2RGB)
        result_img = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
        roi_rgb = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2RGB)

        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        # Plot the images
        axes[0].imshow(target_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(roi_rgb)
        axes[1].set_title('Roi Image')
        axes[1].axis('off')

        axes[2].imshow(result_img)
        axes[2].set_title('Result Image')
        axes[2].axis('off')

        # Show the figure
        plt.show()

        # Display the result and similarity score
        # print(f"Similarity score for {roi_image}: {similarity_score}")
