
                                         # Automated Image Colorization using Image Processing and AI

# Import numpy for numerical operations
import numpy as np
# Import OpenCV for image processing
import cv2
# Import os for path operations
import os
# Import matplotlib for displaying images
import matplotlib.pyplot as plt

# Define paths to load the pre-trained model and other required files

 # Path to the prototxt file
prototxt_path = 'models/colorization_deploy_v2.prototxt'

# Path to the Caffe model
model_path = 'models/colorization_release_v2.caffemodel'

# Path to the numpy file with cluster centers
kernel_path = 'models/pts_in_hull.npy'

# Path to the input black and white image

                                         # change tha image
image_path = 'images/girl.jpeg'
coloured = 'coloured_images/'

# Try to load the colorization model
try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Load the model using the prototxt and model files

except Exception as e:
    print(f"Error loading Caffe model: {e}")
    # Print error message if model loading fails
    exit(1)

# Try to load the cluster centers used by the model
try:
    # Load the cluster centers from the numpy file
    points = np.load(kernel_path)
except Exception as e:
    # Print error message if loading fails
    print(f"Error loading kernel points: {e}")
    # Exit the script
    exit(1)

# Reshape the cluster centers to the shape required by the model
# Transpose and reshape points to (2, 313, 1, 1)
points = points.transpose().reshape(2, 313, 1, 1)
# Assign points to the correct layer
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
# Set the rebalancing factor
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Try to load and process the input black and white image
try:
    # Read the image from the specified path
    bw_image = cv2.imread(image_path)
    if bw_image is None:
        # Raise error if image is not found
        raise ValueError(f"Image not found or unable to read: {image_path}")
except Exception as e:
    # Print error message if loading fails
    print(f"Error loading image: {e}")
    # Exit the script
    exit(1)

# Normalize the image pixel values to the range [0, 1] and convert to LAB color space
# Normalize the image

normalized = bw_image.astype("float32") / 255.0

# Convert image to LAB color space
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

# Resize the image to 224x224 for the model input and normalize the L channel
# Resize the LAB image to 224x224
resized = cv2.resize(lab, (224, 224))

# Extract the L channel
L = cv2.split(resized)[0]
# Subtract 50 to normalize the L channel
L -= 50

# Colorize the image using the model
print("Colorizing the image...")

# Set the L channel as input to the model
net.setInput(cv2.dnn.blobFromImage(L))

# Get the a and b channels from the model's output
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize the a and b channels to the original image size
ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))  # Resize the ab channels to the original image size

# Combine the L channel with the a and b channels
# Extract the L channel from the original LAB image
L = cv2.split(lab)[0]

# Concatenate L with ab channels
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Convert the LAB image back to BGR and scale pixel values to [0, 255]
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

# Convert LAB to BGR color space
colorized = np.clip(colorized, 0, 1)

# Clip pixel values to the range [0, 1]
colorized = (255 * colorized).astype("uint8")

# Scale pixel values to [0, 255] and convert to uint8

# Display the original black and white image and the colorized image using matplotlib
plt.figure(figsize=(10, 5))

# Create a figure of size 10x5 inches

# Create a subplot for the original image
plt.subplot(1, 2, 1)

# Set the title
plt.title("Original")

# Convert BGR to RGB and display the image
plt.imshow(cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB))

# Hide the axis
plt.axis("off")

# Create a subplot for the colorized image
plt.subplot(1, 2, 2)

# Set the title
plt.title("Colorizing AI using Image Processing")

# Convert BGR to RGB and display the image
plt.imshow(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))

# Hide the axis
plt.axis("off")

# Show the figure
plt.show()


# Define the directory for saving colorized images
output_dir = 'coloured_images'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the colorized image in the "coloured_images" folder
output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_colorized.jpg')

cv2.imwrite(output_path, colorized)
print(f"Colorized image saved as: {output_path}")
