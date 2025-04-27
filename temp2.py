import matplotlib.pyplot as plt
import os
import numpy as np

image_folder = r"C:\Users\gonza\Documents\tfg\TFG_testing_code\bias_detection\runs\checkpoint_0_text_run4_apoabp_Person\images"

import matplotlib.image as mpimg

# Get all image file paths in the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Set up the grid dimensions
grid_width = 20
grid_height = 5
# Create a figure
fig, axes = plt.subplots(grid_height, grid_width, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each image in the grid
for i, ax in enumerate(axes):
    if i < len(image_files):
        img = mpimg.imread(image_files[i])
        ax.imshow(img)
        ax.axis('off')  # Hide axes
    else:
        ax.axis('off')  # Hide unused axes

# Remove spacing between images
plt.subplots_adjust(wspace=0, hspace=-0.85)  # Adjust hspace to remove vertical whitespace
plt.savefig("test.png",bbox_inches='tight')
# Show the plot
plt.show()