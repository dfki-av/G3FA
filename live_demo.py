import matplotlib
matplotlib.use('Agg')
import os
import sys
import yaml
from argparse import ArgumentParser
import cv2
import imageio
import numpy as np
from skimage.transform import resize
import torch
import torch.nn.functional as F
from modules.generator import OcclusionAwareGenerator_SPADE
from modules.keypoint_detector import KPDetector
from scipy.spatial import ConvexHull
from typing import Tuple, Union
import math
import mediapipe as mp
from threading import Thread
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  
import threading

mp_face_detection = mp.solutions.face_detection


# VideoGet class for webcam streaming
class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=(), daemon=True).start()
        return self

    def get(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
            if not self.grabbed:
                self.stop()

    def stop(self):
        self.stopped = True
        self.stream.release()

# Define the source images
Source_images = [f"files/avatar{i}.png" for i in range(1, 4)]

# Application state class to avoid global variables
class ApplicationState:
    def __init__(self):
        self.lock = threading.Lock()
        self.source_new = 0
        self.source_updated_flags = {
            'source_updated': True,
            'source_updated5': True,
            'source_updated7': True,
        }
        
    def reset_flags(self):
        for key in self.source_updated_flags:
            self.source_updated_flags[key] = False

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with torch.no_grad():
        # Replace with your actual model classes
        generator = OcclusionAwareGenerator_SPADE(**config['model_params']['generator_params'],
                                                  **config['model_params']['common_params'])
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not cpu:
            generator.cuda()
            kp_detector.cuda()

        if cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)

        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=True,
                 use_relative_movement=True, use_relative_jacobian=True):
    kp_new = {k: v.clone() for k, v in kp_driving.items()}

    # Adapt movement scale based on source and driving areas
    source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
    movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    kp_new['value'] = kp_driving['value'] * movement_scale

    kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
    kp_value_diff *= movement_scale
    kp_new['value'] = kp_value_diff + kp_source['value']

    jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
    kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return 0 <= value <= 1

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(int(normalized_x * image_width), image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def moving_average(current_frame, previous_frame, alpha=0.9):
    keypoint_px_EMA = (
        int(alpha * previous_frame[0][0] + (1 - alpha) * current_frame[0][0]),
        int(alpha * previous_frame[0][1] + (1 - alpha) * current_frame[0][1])
    )
    rect_end_point_EMA = (
        int(alpha * previous_frame[1][0] + (1 - alpha) * current_frame[1][0]),
        int(alpha * previous_frame[1][1] + (1 - alpha) * current_frame[1][1])
    )
    return keypoint_px_EMA, rect_end_point_EMA

# Helper functions
def update_source_from_image(image_index, kp_detector, device):
    source_image = imageio.v2.imread(Source_images[image_index])
    source_image = resize(source_image, (256, 256))[..., :3]
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
    kp_source = kp_detector(source)
    return source, kp_source

def update_source_from_frame(frame, kp_detector, device):
    source_image = resize(frame[:, :, ::-1], (256, 256))[..., :3]
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
    kp_source = kp_detector(source)
    return source, kp_source

def prepare_driving_frame(frame, device):
    driving_video = resize(frame, (256, 256), anti_aliasing=True)[..., :3]
    driving_frame = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
    return driving_frame

def extract_frame(image, keypoint_px_EMA, rect_end_point_EMA):
    # Ensure the coordinates are within image bounds
    h, w = image.shape[:2]
    x_start = max(keypoint_px_EMA[0]-30, 0)
    y_start = max(keypoint_px_EMA[1]-90, 0)
    x_end = min(rect_end_point_EMA[0]+30, w)
    y_end = min(rect_end_point_EMA[1]+30, h)
    frame = image[y_start:y_end, x_start:x_end]
    return frame

def detect_face(image):
    mp_face_detection_module = mp.solutions.face_detection
    with mp_face_detection_module.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rows, image_cols, _ = image.shape
        results = face_detection.process(image_rgb)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            keypoint_px_init = _normalized_to_pixel_coordinates(bbox.xmin, bbox.ymin, image_cols, image_rows)
            rect_end_point_init = _normalized_to_pixel_coordinates(bbox.xmin + bbox.width, bbox.ymin + bbox.height, image_cols, image_rows)
            if keypoint_px_init and rect_end_point_init:
                return keypoint_px_init, rect_end_point_init
    return None, None

# Tkinter GUI function
def start_tkinter_gui(state):
    window = tk.Tk()
    window.title("Button Actions")
    
    label_font = ("Times New Roman", 12)
    label = ttk.Label(window, text="Select an Avatar:", font=label_font)
    label.pack()
    
    style = ttk.Style()
    style.configure("TButton", padding=10, font=("Helvetica", 12))
    
    # Load icons
    icons = []
    icon_indices = [0, 1, 2]
    for idx in icon_indices:
        icon_image = Image.open(Source_images[idx])
        icon_image = icon_image.resize((64, 64))
        icon_photo = ImageTk.PhotoImage(icon_image)
        icons.append(icon_photo)
    
    # Button click handler
    def button_click(value):
        with state.lock:
            state.source_new = value
            state.reset_flags()
            # Set the specific flag to False (un-updated)
            if value == 5:
                state.source_updated_flags['source_updated5'] = False
            elif value == 7:
                state.source_updated_flags['source_updated7'] = False
            else:
                state.source_updated_flags['source_updated'] = False
    
    # Buttons information
    buttons_info = [
        ("Avatar 1", icons[0], lambda: button_click(5)),
        ("Avatar 2", icons[1], lambda: button_click(1)),
        ("Restart", icons[2], lambda: button_click(7))
    ]
    
    # Create buttons
    for text, icon, command in buttons_info:
        button = ttk.Button(window, text=text, image=icon, compound="right", command=command)
        button.image = icon  # Keep a reference
        button.pack(pady=10)
    
    window.mainloop()

# Main animation function
def make_animation(source_image, generator, kp_detector, state,
                   relative=True, adapt_movement_scale=True, estimate_jacobian=False, cpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    keypoint_px_EMA = None
    frame = None
    previous_frame = None
    
    # Create an image with "No Face Detected!" text
    width, height = 256, 256
    image_txt = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_size = 1
    font_thickness = 1
    font_color = (255, 0, 255)
    text_line1 = "No Face"
    text_line2 = "Detected!"
    text_size_line1, _ = cv2.getTextSize(text_line1, font, font_size, font_thickness)
    text_size_line2, _ = cv2.getTextSize(text_line2, font, font_size, font_thickness)
    x_line1 = (width - text_size_line1[0]) // 2
    y_line1 = (height - (text_size_line1[1] + text_size_line2[1])) // 2
    x_line2 = x_line1
    y_line2 = y_line1 + text_size_line1[1] + 10
    cv2.putText(image_txt, text_line1, (x_line1, y_line1), font, font_size, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(image_txt, text_line2, (x_line2, y_line2), font, font_size, font_color, font_thickness, cv2.LINE_AA)
    
    with torch.no_grad():
        # Prepare the source image and keypoints
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        kp_source = kp_detector(source)

        # Initialize webcam stream
        video_getter = VideoGet().start()
        
        # Initialize kp_driving_initial
        driving_frame = None
        kp_driving_initial = None
        
        while True:
            image = video_getter.frame.copy()
            if image is None:
                continue
            
            # Handle source updates
            with state.lock:
                source_new = state.source_new
                source_updated_flags = state.source_updated_flags.copy()
            
            # Update source images or re-initialize keypoints based on user input
            if source_new == 5 and not source_updated_flags['source_updated5']:
                source, kp_source = update_source_from_image(0, kp_detector, device)
                state.source_updated_flags['source_updated5'] = True
            elif source_new == 1 and not source_updated_flags['source_updated']:
                source, kp_source = update_source_from_image(1, kp_detector, device)
                state.source_updated_flags['source_updated'] = True
            elif source_new == 7 and not source_updated_flags['source_updated7']:
                kp_driving_initial = None  # Force re-initialization
                state.source_updated_flags['source_updated7'] = True
            
            # Face detection and keypoint extraction
            keypoint_px_init, rect_end_point_init = detect_face(image)
            if keypoint_px_init is None or rect_end_point_init is None:
                # Display "No Face Detected" message
                numpy_horizontal = np.hstack((image_txt, image_txt))
                enlarged_image = cv2.resize(numpy_horizontal, (1024, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Numpy Horizontal', enlarged_image)
                if cv2.waitKey(1) == 27:
                    break
                continue
            
            # Smooth keypoints using moving average
            if previous_frame is None:
                previous_frame = [keypoint_px_init, rect_end_point_init]
            current_frame = [keypoint_px_init, rect_end_point_init]
            keypoint_px_EMA, rect_end_point_EMA = moving_average(current_frame, previous_frame, alpha=0.9)
            previous_frame = current_frame.copy()
            
            frame = extract_frame(image, keypoint_px_EMA, rect_end_point_EMA)
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                continue
            driving_frame = prepare_driving_frame(frame, device)
            
            if kp_driving_initial is None:
                kp_driving_initial = kp_detector(driving_frame)
            
            # Generate animation frame
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=estimate_jacobian, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            
            # Display results
            numpy_horizontal = np.hstack((driving_frame[0].permute(1, 2, 0).cpu().numpy()[:,:,::-1], predictions))
            enlarged_image = cv2.resize(numpy_horizontal[:, :, ::-1], (1024, 512), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Numpy Horizontal', enlarged_image)
            if cv2.waitKey(1) == 27:
                break

        video_getter.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='ckpt/G3FA.pth', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='files/avatar1.png', help="path to source image")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    opt = parser.parse_args()

    # Read the source image and resize it to 256x256
    source_image = imageio.v2.imread(opt.source_image)
    source_image = resize(source_image, (256, 256))[..., :3]

    # Load the model
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')

    # Initialize application state
    state = ApplicationState()
    
    # Function to start the animation (runs in separate thread)
    def make_animation_wrapper():
        make_animation(
            source_image,
            generator,
            kp_detector,
            state,
            relative=True,
            adapt_movement_scale=True,
            estimate_jacobian=estimate_jacobian,
            cpu=opt.cpu
        )
    
    # Start the animation thread
    animation_thread = threading.Thread(target=make_animation_wrapper)
    animation_thread.start()
    
    # Start the Tkinter GUI (this will block and keep the main thread running)
    start_tkinter_gui(state)
