import cv2
import numpy as np
from scipy.signal import find_peaks

def remove_grayish_background(image, lower_threshold=(0, 0, 150), upper_threshold=(180, 30, 255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    output = np.zeros_like(image)
    output[mask_inv == 255] = image[mask_inv == 255]
    return output


def sharpen_image(image, alpha=1.5, beta=-0.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=5, sigmaY=5)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened


def mask_edges(image, low_threshold=100, high_threshold=200, alpha=1.5, beta=-0.5):
    sharpened_image = sharpen_image(image, alpha, beta)
    gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    kernel = np.ones((10, 10), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    mask_inv = cv2.bitwise_not(edges_dilated)
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(image, mask_inv)
    return result


# def create_bin_mask(clean_image):
#     gray_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
#     # Apply Gaussian blur
#     blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#     # Threshold the image to create a binary image
#     _, binary_mask = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return binary_mask
    

# def get_peak_pixel_val_cords(binary_mask):
#     peaks, _ = find_peaks(binary_mask.sum(-1))
#     return peaks


# def find_median_white_pixel(binary_mask):
#     white_pixels = []
#     for y, row in enumerate(binary_mask):
#         white_indices = np.where(row > 0)[0]
#         for x in white_indices:
#             white_pixels.append(x)

#     if white_pixels:
#         return np.median(white_pixels)
#     else:
#         return None

# def get_final_colors(binary_mask, clean_image, peaks):

#     mid = find_median_white_pixel(binary_mask)    
#     final_colors = []

#     for x in peaks:
#         final_colors.append(clean_image[x][int(mid)])

#     final_colors = np.array(final_colors)[:, ::-1]
#     return final_colors


def map_colors_to_tests(sorted_colors):
    test_keys = ['URO', 'BIL', 'KET', 'BLD', 'PRO', 'NIT', 'LEU', 'GLU', 'SG', 'PH']
    color_map = {key: list(color) for key, color in zip(test_keys, sorted_colors)}
    return color_map


def process_image(image):
    
    clean_image = remove_grayish_background(image, (0, 0, 170), (220, 30, 220))
    # clean_image = sharpen_image(clean_image)
    # clean_image = sharpen_image(clean_image)
    # clean_image = sharpen_image(clean_image)
    clean_image = mask_edges(clean_image, 255, 225, 10, -1.35)
    clean_image = mask_edges(clean_image, 255, 225, 5, -1.35)
    
    # Convert the image to a NumPy array
    image_array = np.array(clean_image)

    # Get the height and width of the image
    height, width, _ = image_array.shape

    # Define a threshold for similarity
    threshold = 10  # You can adjust this value based on your needs

    # Initialize a dictionary to store the row indices based on their mean values
    mean_values_dict = {}

    # Iterate over each row
    for y in range(height):
        # Get the RGB values of each pixel in the row
        row_rgb_values = image_array[y]
        
        # Initialize a list to store the first set of non-black pixels
        first_set_pixels = []
        found_first_pixel = False
        
        for x in range(width):
            rgb = row_rgb_values[x]
            if not np.all(rgb == 0):  # Non-black pixel found
                if not found_first_pixel:
                    found_first_pixel = True
                    first_set_pixels.append(rgb)
                elif found_first_pixel:
                    first_set_pixels.append(rgb)
            elif found_first_pixel:  # Black pixel found after the first set
                break
        
        # Calculate the mean of the first set of non-black pixels
        if first_set_pixels:
            row_mean = tuple(np.round(np.mean(first_set_pixels, axis=0)).astype(int))
            
            # Check if the mean value is already in the dictionary within the threshold
            found = False
            for mean_key in mean_values_dict.keys():
                if np.linalg.norm(np.array(mean_key) - np.array(row_mean)) < threshold:
                    mean_values_dict[mean_key].append(y)
                    found = True
                    break

            # If the mean value is not found within the threshold, add it as a new key
            if not found:
                mean_values_dict[row_mean] = [y]

    # Sort the mean values based on the number of rows they appear in, in descending order
    sorted_mean_values = sorted(mean_values_dict.keys(), key=lambda x: len(mean_values_dict[x]), reverse=True)

    # Ensure to return exactly 10 values, even if some mean values have fewer rows
    top_10_mean_values = sorted_mean_values[:10]

    # Print the top 10 mean RGB values
    # for mean_key in top_10_mean_values:
    #     print(f"Mean RGB: {mean_key}, Rows: {mean_values_dict[mean_key]}")
    return map_colors_to_tests(top_10_mean_values)

# process_image(clean_image)