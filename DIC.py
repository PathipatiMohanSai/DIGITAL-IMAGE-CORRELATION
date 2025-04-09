# py -m pip install opencv-python numpy scikit-image tqdm matplotlib Pillow exifread
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time
import exifread
def get_image_timestamp(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, stop_tag='DateTimeOriginal')
            if 'EXIF DateTimeOriginal' in tags:
                return str(tags['EXIF DateTimeOriginal'])
    except Exception:
        pass
    mod_time = os.path.getmtime(image_path)
    return datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
def load_image(image_path, target_size=(512, 512)):
    try:
        if image_path.lower().endswith(('.tif', '.tiff', 'jpg', 'jpeg', 'png')):
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError
        return cv2.resize(image, target_size)
    except Exception:
        return None
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(processed_image, 30, 100)
    return edges, gray_image
def analyze_crack(args):
    edges_ref, gray_ref, test_image, filename, image_path = args
    if test_image is None:
        return None
    edges_test, gray_test = preprocess_image(test_image)
    if edges_ref.shape != edges_test.shape:
        edges_ref = cv2.resize(edges_ref, (edges_test.shape[1], edges_test.shape[0]))
    if gray_ref.shape != gray_test.shape:
        gray_ref = cv2.resize(gray_ref, (gray_test.shape[1], gray_test.shape[0]))
    edge_diff = cv2.absdiff(edges_ref, edges_test)
    kernel = np.ones((5, 5), np.uint8)
    edge_diff_enhanced = cv2.dilate(edge_diff, kernel, iterations=1)
    _, edge_diff_thresh = cv2.threshold(edge_diff_enhanced, 50, 255, cv2.THRESH_BINARY)
    crack_area = cv2.countNonZero(edge_diff_thresh)
    score = ssim(gray_ref, gray_test, full=False)
    timestamp = get_image_timestamp(image_path)
    return {'image': test_image, 'filename': filename, 'crack_area': crack_area,
            'ssim_score': score, 'timestamp': timestamp}
def detect_largest_ssim_change(reference_path, images_folder, target_size=(512, 512)):
    start_time = time.time()
    reference_image = load_image(reference_path, target_size)
    if reference_image is None:
        raise ValueError(f"Reference image at {reference_path} could not be loaded.")
    supported_formats = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    image_files = sorted([f for f in os.listdir(images_folder) if f.lower().endswith(supported_formats)])
    if not image_files:
        raise ValueError(f"No images found in the folder: {images_folder}")
    print(f"Analyzing {len(image_files)} images...")
    edges_ref, gray_ref = preprocess_image(reference_image)
    crack_data = []
    with ThreadPoolExecutor() as executor:
        tasks = [(edges_ref, gray_ref, load_image(os.path.join(images_folder, f), target_size),
                 f, os.path.join(images_folder, f)) for f in image_files]
        results = list(tqdm(executor.map(analyze_crack, tasks), total=len(image_files), desc="Processing Images"))
    for result in results:
        if result is not None:
            crack_data.append(result)
            print(f"Analyzed {result['filename']}: Crack Area = {result['crack_area']}, "
                  f"SSIM Score = {result['ssim_score']:.4f}, Timestamp = {result['timestamp']}")
    max_ssim_diff = 0
    selected_image = None
    selected_filename = None
    selected_crack_area = 0
    selected_score = 1.0
    selected_timestamp = None
    for i in range(1, len(crack_data)):
        prev_score = crack_data[i - 1]['ssim_score']
        curr_score = crack_data[i]['ssim_score']
        ssim_diff = abs(prev_score - curr_score)
        if ssim_diff > max_ssim_diff:
            max_ssim_diff = ssim_diff
            selected_image = crack_data[i]['image']
            selected_filename = crack_data[i]['filename']
            selected_crack_area = crack_data[i]['crack_area']
            selected_score = crack_data[i]['ssim_score']
            selected_timestamp = crack_data[i]['timestamp']
    if max_ssim_diff < 0.005 and crack_data:
        print("No crack detected: SSIM difference is less than 0.005")
        selected_image = crack_data[-1]['image']
        selected_filename = crack_data[-1]['filename']
        selected_crack_area = crack_data[-1]['crack_area']
        selected_score = crack_data[-1]['ssim_score']
        selected_timestamp = crack_data[-1]['timestamp']
    elif selected_image is None and crack_data:
        selected_image = crack_data[-1]['image']
        selected_filename = crack_data[-1]['filename']
        selected_crack_area = crack_data[-1]['crack_area']
        selected_score = crack_data[-1]['ssim_score']
        selected_timestamp = crack_data[-1]['timestamp']
        print("No significant SSIM change detected. Defaulting to the last image.")
    print(f"Largest SSIM difference detected: {max_ssim_diff:.4f}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return selected_image, selected_filename, selected_score, selected_crack_area, selected_timestamp
def display_results(image, filename, score, crack_area, timestamp):
    if image is None:
        print("No suitable image detected.")
        return
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Largest SSIM Change Detected\nFilename: {filename}\nSSIM Score: {score:.4f}\n"
              f"Crack Area: {crack_area} pixels\nImage Timestamp: {timestamp}\n"
              f"Analysis Timestamp: {current_time}")
    plt.axis('off')
    plt.show()
    print(f"\nResults:")
    print(f"Image with largest SSIM change from previous: {filename}")
    print(f"SSIM Score: {score:.4f}")
    print(f"Crack Area: {crack_area} pixels")
    print(f"Image Timestamp: {timestamp}")
    print(f"Analysis Timestamp: {current_time}")
if __name__ == "__main__":
    reference_image_path = "reference_image.tif"
    images_folder_path = "New folder"
    try:
        detected_image, detected_filename, detected_score, detected_crack_area, detected_timestamp = detect_largest_ssim_change(
            reference_image_path, images_folder_path
        )
        display_results(detected_image, detected_filename, detected_score, detected_crack_area, detected_timestamp)
    except Exception as e:
        print(f"An error occurred: {str(e)}")