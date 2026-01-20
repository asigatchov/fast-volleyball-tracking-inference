#!/usr/bin/env python3
"""
Volleyball court keypoints annotation tool
Allows manual annotation of volleyball court keypoints on images
"""
import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod


class FrameSource(ABC):
    """Abstract base class for frame sources (directory or video)"""
    
    @abstractmethod
    def get_frame(self, index):
        """Get frame at specified index"""
        pass
    
    @abstractmethod
    def get_total_frames(self):
        """Get total number of frames"""
        pass
    
    @abstractmethod
    def get_frame_name(self, index):
        """Get frame name at specified index"""
        pass
    
    @abstractmethod
    def save_frame(self, index, frame):
        """Save frame at specified index"""
        pass
    
    @abstractmethod
    def save_annotation(self, index, keypoints):
        """Save annotation for frame at specified index"""
        pass
    
    @abstractmethod
    def delete_frame(self, index):
        """Delete frame and annotation at specified index"""
        pass
    
    @abstractmethod
    def get_annotation_path(self, index):
        """Get path to annotation file for frame at specified index"""
        pass


class DirectoryFrameSource(FrameSource):
    """Frame source for directory containing image files"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_files = []
        
        # Find all image files in the data directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in image_extensions:
            self.image_files.extend(list(self.data_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.data_dir.glob(f'*{ext.upper()}')))
        
        # Sort files for consistent ordering
        self.image_files = sorted(self.image_files)
        

    def get_frame(self, index):
        """Get frame at specified index"""
        if index < 0 or index >= len(self.image_files):
            return None
        image_path = self.image_files[index]
        return cv2.imread(str(image_path))
    
    def get_total_frames(self):
        """Get total number of frames"""
        return len(self.image_files)
    
    def get_frame_name(self, index):
        """Get frame name at specified index"""
        if index < 0 or index >= len(self.image_files):
            return None
        return self.image_files[index].name
    
    def save_frame(self, index, frame):
        """Save frame at specified index - not needed for directory source"""
        # For directory source, frame is already in place
        pass
    
    def save_annotation(self, index, keypoints):
        """Save annotation for frame at specified index"""
        if index < 0 or index >= len(self.image_files):
            return
            
        image_path = self.image_files[index]
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        
        # Convert keypoints to flat list [x1,y1,v1,x2,y2,v2,...]
        keypoints_flat = []
        for x, y, v in keypoints:
            keypoints_flat.extend([x, y, v])
        
        # Create the COCO structure
        coco_data = {
            "images": [{
                "id": 0,
                "file_name": image_path.name,
                "width": width,
                "height": height
            }],
            "annotations": [{
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "keypoints": keypoints_flat,
                "num_keypoints": sum(1 for x, y, v in keypoints if v > 0),
                "bbox": self.calculate_bbox(keypoints_flat)
            }],
            "categories": [{
                "id": 0,
                "name": "volleyball_court",
                "keypoints": [
                    "1_back_left",      # 0
                    "2_back_left",      # 1
                    "3_back_right",     # 2
                    "4_back_right",     # 3
                    "5_center_left",    # 4
                    "6_center_right",   # 5
                    "7_net_left",       # 6
                    "8_net_right"       # 7
                ],
                "skeleton": [
                    [0, 4], [4, 1], [1, 2], 
                    [2, 5], [5, 3], [3, 0],
                    [4,5],[4,6],[6,7],[7,5]
                ]
            }]
        }
        
        json_path = image_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def delete_frame(self, index):
        """Delete frame and annotation at specified index"""
        if index < 0 or index >= len(self.image_files):
            return
            
        image_path = self.image_files[index]
        json_path = image_path.with_suffix('.json')
        
        # Delete the JSON annotation file if it exists
        if json_path.exists():
            json_path.unlink()
        
        # Delete the image file
        image_path.unlink()
        
        # Remove the image from our list
        self.image_files.pop(index)
    
    def get_annotation_path(self, index):
        """Get path to annotation file for frame at specified index"""
        if index < 0 or index >= len(self.image_files):
            return None
        return self.image_files[index].with_suffix('.json')
    
    def calculate_bbox(self, keypoints_flat):
        """Calculate bounding box from keypoints [x1,y1,v1,x2,y2,v2,...]"""
        visible_points = []
        for i in range(0, len(keypoints_flat), 3):
            x, y, v = keypoints_flat[i], keypoints_flat[i+1], keypoints_flat[i+2]
            if v > 0:  # Only visible points
                visible_points.append([x, y])
        
        if not visible_points:
            return [0, 0, 1, 1]  # Default small bbox
        
        visible_points = np.array(visible_points)
        min_x = int(np.min(visible_points[:, 0]))
        min_y = int(np.min(visible_points[:, 1]))
        max_x = int(np.max(visible_points[:, 0]))
        max_y = int(np.max(visible_points[:, 1]))
        
        width = max_x - min_x
        height = max_y - min_y
        
        return [min_x, min_y, width, height]


class VideoFrameSource(FrameSource):
    """Frame source for video files"""
    
    def __init__(self, video_path, frame_step=30, target_width=1280, mirror=False):
        self.video_path = Path(video_path)
        self.frame_step = frame_step
        self.target_width = target_width
        self.mirror = mirror
        self.video_name = self.video_path.stem
        if self.mirror:
            self.output_dir = self.video_path.parent / f"{self.video_name}_mirror_frames"
        else:
            self.output_dir = self.video_path.parent / f"{self.video_name}_frames"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of frame numbers to process
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video {self.video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Collect frames to process (every frame_step-th frame)
        self.frames_to_process = []
        temp_frame_count = 0
        while temp_frame_count < self.total_frames:
            if temp_frame_count % self.frame_step == 0:
                self.frames_to_process.append(temp_frame_count)
            temp_frame_count += self.frame_step
        
        # Get actual image files in the output directory
        self.image_files = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in image_extensions:
            self.image_files.extend(list(self.output_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.output_dir.glob(f'*{ext.upper()}')))
        
        # Sort files for consistent ordering
        self.image_files = sorted(self.image_files)

    def get_frame(self, index):
        """Get frame at specified index"""
        if index < 0 or index >= len(self.frames_to_process):
            return None
        
        frame_num = self.frames_to_process[index]
        
        # Read the specific frame from video
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Resize frame to target width while maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = self.target_width / w
        new_height = int(h * scale)
        resized_frame = cv2.resize(frame, (self.target_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Mirror frame if requested (horizontal flip)
        if self.mirror:
            resized_frame = cv2.flip(resized_frame, 1)  # 1 means horizontal flip
        
        return resized_frame
    
    def get_total_frames(self):
        """Get total number of frames"""
        return len(self.frames_to_process)
    
    def get_frame_name(self, index):
        """Get frame name at specified index"""
        if index < 0 or index >= len(self.frames_to_process):
            return None
        
        frame_num = self.frames_to_process[index]
        video_name = self.video_path.stem
        if self.mirror:
            video_name = f"{video_name}_mirror"
        
        return f"{video_name}_frame_{frame_num:06d}.jpg"
    
    def save_frame(self, index, frame):
        """Save frame at specified index"""
        if index < 0 or index >= len(self.frames_to_process):
            return
        
        frame_num = self.frames_to_process[index]
        video_name = self.video_path.stem
        if self.mirror:
            video_name = f"{video_name}_mirror"
        
        frame_filename = f"{video_name}_frame_{frame_num:06d}.jpg"
        frame_path = self.output_dir / frame_filename
        
        cv2.imwrite(str(frame_path), frame)
    
    def save_annotation(self, index, keypoints):
        """Save annotation for frame at specified index"""
        if index < 0 or index >= len(self.frames_to_process):
            return
        
        frame_num = self.frames_to_process[index]
        video_name = self.video_path.stem
        if self.mirror:
            video_name = f"{video_name}_mirror"
        
        # Get the frame to determine dimensions
        frame = self.get_frame(index)
        if frame is None:
            return
        
        height, width = frame.shape[:2]

        # Convert keypoints to flat list [x1,y1,v1,x2,y2,v2,...]
        keypoints_flat = []
        for x, y, v in keypoints:
            keypoints_flat.extend([x, y, v])

        # Create the COCO structure
        coco_data = {
            "images": [{
                "id": 0,
                "file_name": self.get_frame_name(index),
                "width": width,
                "height": height
            }],
            "annotations": [{
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "keypoints": keypoints_flat,
                "num_keypoints": sum(1 for x, y, v in keypoints if v > 0),
                "bbox": self.calculate_bbox(keypoints_flat)
            }],
            "categories": [{
                "id": 0,
                "name": "volleyball_court",
                "keypoints": [
                    "1_back_left",      # 0
                    "2_back_left",      # 1
                    "3_back_right",     # 2
                    "4_back_right",     # 3
                    "5_center_left",    # 4
                    "6_center_right",   # 5
                    "7_net_left",       # 6
                    "8_net_right"       # 7
                ],
                "skeleton": [
                    [0, 4], [4, 1], [1, 2], 
                    [2, 5], [5, 3], [3, 0],
                    [4,5],[4,6],[6,7],[7,5]
                ]
            }]
        }
        
        frame_name = self.get_frame_name(index)
        json_filename = frame_name.replace('.jpg', '.json')
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def delete_frame(self, index):
        """Delete frame and annotation at specified index"""
        if index < 0 or index >= len(self.frames_to_process):
            return
        
        frame_num = self.frames_to_process[index]
        video_name = self.video_path.stem
        if self.mirror:
            video_name = f"{video_name}_mirror"
        
        frame_filename = f"{video_name}_frame_{frame_num:06d}.jpg"
        json_filename = f"{video_name}_frame_{frame_num:06d}.json"
        
        frame_path = self.output_dir / frame_filename
        json_path = self.output_dir / json_filename
        
        # Delete the JSON annotation file if it exists
        if json_path.exists():
            json_path.unlink()
        
        # Delete the image file
        if frame_path.exists():
            frame_path.unlink()
        
        # Remove from our list if it exists
        if frame_path in self.image_files:
            self.image_files.remove(frame_path)
    
    def get_annotation_path(self, index):
        """Get path to annotation file for frame at specified index"""
        if index < 0 or index >= len(self.frames_to_process):
            return None
        
        frame_num = self.frames_to_process[index]
        video_name = self.video_path.stem
        if self.mirror:
            video_name = f"{video_name}_mirror"
        
        json_filename = f"{video_name}_frame_{frame_num:06d}.json"
        return self.output_dir / json_filename
    
    def calculate_bbox(self, keypoints_flat):
        """Calculate bounding box from keypoints [x1,y1,v1,x2,y2,v2,...]"""
        visible_points = []
        for i in range(0, len(keypoints_flat), 3):
            x, y, v = keypoints_flat[i], keypoints_flat[i+1], keypoints_flat[i+2]
            if v > 0:  # Only visible points
                visible_points.append([x, y])
        
        if not visible_points:
            return [0, 0, 1, 1]  # Default small bbox
        
        visible_points = np.array(visible_points)
        min_x = int(np.min(visible_points[:, 0]))
        min_y = int(np.min(visible_points[:, 1]))
        max_x = int(np.max(visible_points[:, 0]))
        max_y = int(np.max(visible_points[:, 1]))
        
        width = max_x - min_x
        height = max_y - min_y
        
        return [min_x, min_y, width, height]

class VolleyballCourtAnnotator:
    def __init__(self, frame_source):
        self.frame_source = frame_source
        self.current_image_idx = 0
        self.current_image = None
        self.current_annotations = None
        
        # Volleyball court keypoints (8 points)
        self.keypoints_names = [
            "1_back_left",      # 0
            "2_back_left",      # 1
            "3_back_right",     # 2
            "4_back_right",     # 3
            "5_center_left",    # 4
            "6_center_right",   # 5
            "7_net_left",       # 6
            "8_net_right"       # 7
        ]
        
        # Keypoint colors (BGR format) - distinct and contrasting
        self.keypoint_colors = [
            (0, 0, 255),    # 0 - Red
            (0, 165, 255),  # 1 - Orange
            (0, 255, 255),  # 2 - Yellow
            (0, 255, 0),    # 3 - Green
            (255, 0, 0),    # 4 - Blue
            (128, 0, 128),  # 5 - Purple
            (0, 255, 255),  # 6 - Cyan
            (128, 0, 128)   # 7 - Magenta
        ]
        
        # Different colors for visible vs not visible
        self.visible_color = (0, 255, 0)    # Green
        self.not_visible_color = (0, 0, 255) # Red
        
        self.current_keypoint = 0
        # Initialize all keypoints with visibility 0 (not visible) at start
        self.keypoints = [(0, 0, 0) for _ in range(8)]  # List of (x, y, visibility) for each keypoint
        self.drawing = False
        
        # Initialize keypoints buffer for all images
        self.keypoints_buffer = {}  # Dictionary to store keypoints for each image
        
        # Clipboard buffer for copy/paste operations
        self.clipboard_buffer = None  # Store keypoints for copy/paste between images
        
        if self.frame_source.get_total_frames() == 0:
            print(f"No frames found in source")
            sys.exit(1)
        
        print(f"Found {self.frame_source.get_total_frames()} frames")
    
    def load_annotations(self):
        """Load existing annotations from JSON file if available, or from buffer"""
        if self.current_image_idx < self.frame_source.get_total_frames():
            # Get frame name for buffer key
            frame_name = self.frame_source.get_frame_name(self.current_image_idx)
            if frame_name:
                frame_path_str = str(self.frame_source.output_dir / frame_name) if hasattr(self.frame_source, 'output_dir') else frame_name
            else:
                frame_path_str = f"frame_{self.current_image_idx}"
            
            # Check if we have annotations in buffer for this frame
            if frame_path_str in self.keypoints_buffer:
                # Load from buffer
                self.keypoints = self.keypoints_buffer[frame_path_str].copy()
                print(f"Loaded annotations from buffer for {frame_name}")
            else:
                # Try to load from JSON file
                json_path = self.frame_source.get_annotation_path(self.current_image_idx)
                if json_path and json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            # Look for the annotation for this specific image
                            if 'annotations' in data and len(data['annotations']) > 0:
                                # Get the first annotation
                                ann = data['annotations'][0]
                                # keypoints format is [x1,y1,v1,x2,y2,v2,...]
                                kp_values = ann.get('keypoints', [])
                                # Convert to our format [(x, y, v), ...]
                                self.keypoints = []
                                for i in range(0, len(kp_values), 3):
                                    if i + 2 < len(kp_values):
                                        self.keypoints.append((kp_values[i], kp_values[i+1], kp_values[i+2]))
                                    else:
                                        self.keypoints.append((0, 0, 0))  # Default if incomplete
                                print(f"Loaded existing annotations from {json_path}")
                                # Also store in buffer
                                self.keypoints_buffer[frame_path_str] = self.keypoints.copy()
                            else:
                                # Initialize with empty keypoints
                                self.keypoints = [(0, 0, 0) for _ in range(8)]
                                # Initialize buffer for this frame
                                self.keypoints_buffer[frame_path_str] = self.keypoints.copy()
                        return True
                    except Exception as e:
                        print(f"Error loading annotations: {e}")
                else:
                    # Initialize with empty keypoints if no existing annotations
                    self.keypoints = [(0, 0, 0) for _ in range(8)]
                    # Initialize buffer for this frame
                    self.keypoints_buffer[frame_path_str] = self.keypoints.copy()
                    return False
        else:
            # Initialize with empty keypoints if no existing annotations
            self.keypoints = [(0, 0, 0) for _ in range(8)]
            return False
    
    def save_annotations(self):
        """Save annotations in COCO format from buffer"""
        if self.current_image_idx >= self.frame_source.get_total_frames():
            return
            
        # Get the frame to determine dimensions
        frame = self.frame_source.get_frame(self.current_image_idx)
        if frame is None:
            return
        height, width = frame.shape[:2]
        
        # Get keypoints from buffer for this frame
        frame_name = self.frame_source.get_frame_name(self.current_image_idx)
        frame_path_str = str(self.frame_source.output_dir / frame_name) if hasattr(self.frame_source, 'output_dir') else frame_name
        
        if frame_path_str in self.keypoints_buffer:
            keypoints_to_save = self.keypoints_buffer[frame_path_str]
        else:
            keypoints_to_save = self.keypoints
        
        # Save the frame and annotation using the frame source
        self.frame_source.save_frame(self.current_image_idx, frame)
        self.frame_source.save_annotation(self.current_image_idx, keypoints_to_save)
        
        print(f"Saved annotations for frame {frame_name}")
    
    def calculate_bbox(self, keypoints_flat):
        """Calculate bounding box from keypoints [x1,y1,v1,x2,y2,v2,...]"""
        visible_points = []
        for i in range(0, len(keypoints_flat), 3):
            x, y, v = keypoints_flat[i], keypoints_flat[i+1], keypoints_flat[i+2]
            if v > 0:  # Only visible points
                visible_points.append([x, y])
        
        if not visible_points:
            return [0, 0, 1, 1]  # Default small bbox
        
        visible_points = np.array(visible_points)
        min_x = int(np.min(visible_points[:, 0]))
        min_y = int(np.min(visible_points[:, 1]))
        max_x = int(np.max(visible_points[:, 0]))
        max_y = int(np.max(visible_points[:, 1]))
        
        width = max_x - min_x
        height = max_y - min_y
        
        return [min_x, min_y, width, height]
    
    def copy_keypoints(self):
        """Copy current keypoints to clipboard buffer"""
        # Copy the current keypoints to the clipboard buffer
        self.clipboard_buffer = self.keypoints.copy()
        frame_name = self.frame_source.get_frame_name(self.current_image_idx)
        print(f"Keypoints copied to clipboard buffer. Current frame: {frame_name}")
    
    def paste_keypoints(self):
        """Paste keypoints from clipboard buffer to current image"""
        if self.clipboard_buffer is not None:
            # Apply the clipboard buffer to current keypoints
            self.keypoints = self.clipboard_buffer.copy()
            # Update buffer with the pasted keypoints
            frame_name = self.frame_source.get_frame_name(self.current_image_idx)
            frame_path_str = str(self.frame_source.output_dir / frame_name) if hasattr(self.frame_source, 'output_dir') else frame_name
            self.keypoints_buffer[frame_path_str] = self.keypoints.copy()
            print(f"Keypoints pasted from clipboard buffer to current frame: {frame_name}")
            # Redraw the image to show the pasted keypoints immediately
            if self.current_image is not None:
                annotated_img = self.draw_annotations(self.current_image)
                cv2.imshow('Volleyball Court Annotation', annotated_img)
        else:
            print("Clipboard buffer is empty. Nothing to paste.")
    
    def delete_current_image(self):
        """Delete current image and its associated JSON annotation file"""
        if self.current_image_idx < self.frame_source.get_total_frames():
            # Use frame source to delete the frame
            self.frame_source.delete_frame(self.current_image_idx)
            
            # Remove from keypoints buffer
            frame_name = self.frame_source.get_frame_name(self.current_image_idx)
            frame_path_str = str(self.frame_source.output_dir / frame_name) if hasattr(self.frame_source, 'output_dir') else frame_name
            if frame_path_str in self.keypoints_buffer:
                del self.keypoints_buffer[frame_path_str]
            
            # Adjust current index if needed - for now just move to next frame
            if self.current_image_idx >= self.frame_source.get_total_frames() and self.current_image_idx > 0:
                self.current_image_idx = self.frame_source.get_total_frames() - 1
            elif self.current_image_idx >= self.frame_source.get_total_frames():
                # If we're at the end and no more frames, we're done
                if self.frame_source.get_total_frames() == 0:
                    print("All frames deleted. Exiting...")
                    cv2.destroyAllWindows()
                    exit()
            
            print(f"Deleted frame and annotation")
            
            # Reload the current image
            if self.current_image_idx < self.frame_source.get_total_frames():
                self.load_annotations()
                # Show the next image
                self.current_image = self.frame_source.get_frame(self.current_image_idx)
                annotated_img = self.draw_annotations(self.current_image)
                window_title = f'Volleyball Court Annotation - {self.frame_source.get_frame_name(self.current_image_idx)}'
                cv2.imshow('Volleyball Court Annotation', annotated_img)
                cv2.setWindowTitle('Volleyball Court Annotation', window_title)
            else:
                print("No more frames to display")
                cv2.destroyAllWindows()
                exit()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for handling clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click to set current keypoint
            self.keypoints[self.current_keypoint] = (x, y, 2)  # 2 = labeled and visible
            # Update buffer with the new keypoint
            frame_name = self.frame_source.get_frame_name(self.current_image_idx)
            frame_path_str = str(self.frame_source.output_dir / frame_name) if hasattr(self.frame_source, 'output_dir') else frame_name
            self.keypoints_buffer[frame_path_str] = self.keypoints.copy()
            print(f"Set keypoint '{self.keypoints_names[self.current_keypoint]}' at ({x}, {y})")
            # Redraw the image to show the new keypoint immediately
            if self.current_image is not None:
                annotated_img = self.draw_annotations(self.current_image)
                cv2.imshow('Volleyball Court Annotation', annotated_img)
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle click to set current keypoint visibility to 0 with coordinates (0, 0)
            self.keypoints[self.current_keypoint] = (0, 0, 0)  # Set to coordinates (0, 0) with visibility 0
            # Update buffer with the modified keypoint
            frame_name = self.frame_source.get_frame_name(self.current_image_idx)
            frame_path_str = str(self.frame_source.output_dir / frame_name) if hasattr(self.frame_source, 'output_dir') else frame_name
            self.keypoints_buffer[frame_path_str] = self.keypoints.copy()
            print(f"Set keypoint '{self.keypoints_names[self.current_keypoint]}' to invisible at (0, 0)")
            # Redraw the image to show the change immediately
            if self.current_image is not None:
                annotated_img = self.draw_annotations(self.current_image)
                cv2.imshow('Volleyball Court Annotation', annotated_img)
    
    def draw_annotations(self, image):
        """Draw keypoints and connections on image"""
        img_copy = image.copy()
        
        # Draw skeleton connections
        skeleton = [
            [0, 4], [4, 1], [1, 2], 
            [2, 5], [5, 3], [3, 0],
            [4,5],[4,6],[6,7],[7,5]
        ]
        
        for conn in skeleton:
            pt1_idx, pt2_idx = conn
            pt1 = self.keypoints[pt1_idx]
            pt2 = self.keypoints[pt2_idx]
            
            # Only draw if both points are visible (v > 0)
            if pt1[2] > 0 and pt2[2] > 0:  # Both visible
                cv2.line(img_copy, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), 
                        (200, 200, 200), 2)
        
        # Draw keypoints
        for i, (x, y, v) in enumerate(self.keypoints):
            if v > 0:  # Draw visible points
                color = self.visible_color if v == 2 else self.not_visible_color
                # Draw point
                cv2.circle(img_copy, (int(x), int(y)), 8, color, -1)
                cv2.circle(img_copy, (int(x), int(y)), 8, (255, 255, 255), 2)  # White border
                
                # Draw keypoint number
                cv2.putText(img_copy, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img_copy, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:  # Draw invisible points as smaller, semi-transparent circles
                # Draw as smaller, gray circle with dashed border
                cv2.circle(img_copy, (int(x), int(y)), 4, (128, 128, 128), 1, lineType=cv2.LINE_8)
                # Draw keypoint number in gray
                cv2.putText(img_copy, str(i), (int(x)+6, int(y)-6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Highlight current keypoint
        if self.current_keypoint < len(self.keypoints):
            curr_x, curr_y, curr_v = self.keypoints[self.current_keypoint]
            # Always highlight the current keypoint regardless of visibility
            cv2.circle(img_copy, (int(curr_x), int(curr_y)), 12, (0, 255, 255), 3)  # Yellow highlight
        
        # Add info text
        h, w = img_copy.shape[:2]
        cv2.putText(img_copy, f"Frame: {self.current_image_idx + 1}/{self.frame_source.get_total_frames()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Frame: {self.current_image_idx + 1}/{self.frame_source.get_total_frames()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        cv2.putText(img_copy, f"Keypoint: {self.current_keypoint} - {self.keypoints_names[self.current_keypoint]}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Keypoint: {self.current_keypoint} - {self.keypoints_names[self.current_keypoint]}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add instructions
        instructions = [
            "Controls:",
            "LMB - Set current keypoint",
            "MMB - Delete current keypoint",
            "N - Next keypoint",
            "P - Previous keypoint", 
            "SPACE - Skip keypoint",
            "S - Save annotations",
            "A - Auto mode (next frame + paste + save)",
            "D - Delete current image and annotation",
            "Ctrl+C - Copy keypoints",
            "Ctrl+V - Paste keypoints",
            '"]" - Next image',
            '"[" - Previous image',
            "ESC - Quit"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(img_copy, text, (w - 250, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img_copy, text, (w - 250, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_copy

    def run(self):
        """Main annotation loop"""
        cv2.namedWindow('Volleyball Court Annotation', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Volleyball Court Annotation', self.mouse_callback)
        
        fps = 0
        while True:
            # Load current frame
            if self.current_image_idx < self.frame_source.get_total_frames():
                self.current_image = self.frame_source.get_frame(self.current_image_idx)
                
                if self.current_image is None:
                    print(f"Could not load frame: {self.current_image_idx}")
                    continue
                
                # Load existing annotations or initialize
                self.load_annotations()
                
                # Draw annotations on frame
                annotated_img = self.draw_annotations(self.current_image)
                
                # Set window title to current file name
                window_title = f'Volleyball Court Annotation - {self.frame_source.get_frame_name(self.current_image_idx)}'
                cv2.imshow('Volleyball Court Annotation', annotated_img)
                cv2.setWindowTitle('Volleyball Court Annotation', window_title)
                
                # Wait for key press
                key = cv2.waitKey(fps) & 0xFF
                
                if key == 27:  # ESC key
                    break
                elif key == ord('n') or key == ord('N'):  # Next keypoint
                    fps = 0
                    self.current_keypoint = (self.current_keypoint + 1) % 8
                elif key == ord('p') or key == ord('P'):  # Previous keypoint
                    self.current_keypoint = (self.current_keypoint - 1) % 8
                elif key == ord(']'):  # Next frame
                    if self.current_image_idx < self.frame_source.get_total_frames() - 1:
                        self.current_image_idx += 1
                elif key == ord('['):  # Previous frame
                    if self.current_image_idx > 0:
                        self.current_image_idx -= 1
                elif key == ord(' '):  # Space - skip keypoint
                    # Mark current keypoint as not visible (0) with coordinates (0, 0)
                    if self.current_keypoint < len(self.keypoints):
                        self.keypoints[self.current_keypoint] = (0, 0, 0)
                        print(f"Skipped keypoint '{self.keypoints_names[self.current_keypoint]}'")
                        self.current_keypoint = (self.current_keypoint + 1) % 8
                elif key == ord('s') or key == ord('S'):  # Save
                    self.save_annotations()
                elif key == ord('m') or key == ord('M'):  # Toggle mode (keypoint/frame navigation)
                    # This would toggle between navigating keypoints vs frames
                    # For now, let's just add a trackbar to distinguish modes
                    pass
                elif key == ord('c') or key == ord('C'):  # Ctrl+C - Copy keypoints to clipboard buffer
                    self.copy_keypoints()
                elif key == ord('v') or key == ord('V'):  # Ctrl+V - Paste keypoints from clipboard buffer
                    self.paste_keypoints()
                elif key == ord('a') or key == ord('A'):  # Auto mode - go to next frame, paste keypoints from buffer, and save
                    # Go to next frame
                    fps = 0
                    if self.current_image_idx < self.frame_source.get_total_frames() - 1:
                        self.current_image_idx += 1
                        # Paste keypoints from clipboard buffer if available
                        if self.clipboard_buffer is not None:
                            fps = 1
                            self.paste_keypoints()
                            # Save the annotations
                            self.save_annotations()
                            print(f"Auto mode: Navigated to next frame, pasted keypoints and saved")
                        else:
                            print(f"Auto mode: Navigated to next frame, but clipboard buffer is empty")
                    else:
                        print(f"Auto mode: Already at the last frame")
                elif key == ord('d') or key == ord('D'):  # Delete current frame and annotation
                    self.delete_current_image()

                if fps != 0 and self.clipboard_buffer is not None:
                    self.current_image_idx += 1
                    self.paste_keypoints()
                    # Save the annotations
                    self.save_annotations()
                    print(f"Auto mode: Navigated to next frame, pasted keypoints and saved")

            else:
                break
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Volleyball Court Keypoints Annotation Tool')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing images to annotate (default: data)')
    parser.add_argument('--video_path', type=str, 
                       help='Path to video file to extract frames from')
    parser.add_argument('--frame_step', type=int, default=30,
                       help='Step between frames when extracting from video (default: 30)')
    parser.add_argument('--target_width', type=int, default=1280,
                       help='Target width for resized frames (default: 1280)')
    parser.add_argument('--mirror', action='store_true',
                       help='Horizontally flip frames (mirror effect)')
    
    args = parser.parse_args()
    
    # Create the appropriate frame source based on arguments
    if args.video_path:
        video_path = Path(args.video_path)
        if not video_path.exists():
            print(f"Error: Video file {video_path} does not exist")
            return
        
        # Create VideoFrameSource
        frame_source = VideoFrameSource(
            video_path=str(video_path),
            frame_step=args.frame_step,
            target_width=args.target_width,
            mirror=args.mirror
        )
    else:
        # Create DirectoryFrameSource
        frame_source = DirectoryFrameSource(args.data_dir)
    
    # Create and run the annotator with the frame source
    annotator = VolleyballCourtAnnotator(frame_source)
    annotator.run()

if __name__ == "__main__":
    main()