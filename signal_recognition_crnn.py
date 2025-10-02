from collections import defaultdict, deque

import cv2
from PIL import Image
# from signal_recognition import * # type: ignore
from ultralytics import YOLO # type: ignore

from crnn.predict import ResNetLSTMPredictor
from utils.utils import *


class VehicleSignalRecognizer:
    def __init__(self, config):
        # Configuration
        self.CONF_THRESHOLD = config.get('conf_threshold', 0.8)
        self.SKIP_TIME = config.get('skip_time', 0.2)
        self.FRAME_PER_SECOND = 1 / self.SKIP_TIME
        self.MIN_FRAMES_PER_OBJECT = config.get('min_frames', 10)
        self.MAX_FRAMES_PER_OBJECT = config.get('max_frames', 15)
                
        # Load models
        self.vehicle_detector = YOLO(config['vehicle_model_path'])
        self.signal_recognitor = ResNetLSTMPredictor(config['signal_model_config'])
        
        # Load polygon
        self.polygon = load_polygon(config['polygon_path'])
        
        # Load line check
        self.line_check = load_polygon(config['line_check_path'])
        
        # Tracking data
        # self.vehicles = defaultdict(list)
        self.vehicles = defaultdict(lambda: {
            'frames': deque(maxlen=self.MAX_FRAMES_PER_OBJECT),
            'under_line': False,
            'inside_polygon': False,
            'if_track': False,
            'other_meta': None,
        })
        self.signal_text = defaultdict(tuple)
        self.last_seen_frame = {}  # Track when each vehicle was last seen
        
        # Video properties
        self.cap = None
        self.out = None
        self.fps = None
        self.interval = None
        self.frame_count = 0
        
        # Pre-define colors for better performance
        self.colors = {
            'polygon': (0, 200, 0),
            'line_check': (200, 0, 0),
            'vehicle_box': (0, 180, 255),
            'text_bg': (0, 180, 255),
            'text': (255, 255, 255),
            'left_signal': (0, 180, 0),
            'right_signal': (180, 0, 0),
            'no_signal': (80, 80, 80)
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1
        

    def setup_video(self, input_path, output_path):
        """Setup video capture and writer"""
        self.cap = cv2.VideoCapture(input_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        self.interval = int(self.fps * self.SKIP_TIME)

    def draw_text_with_background(self, image, text, position, bg_color):
        """Draw text with background more efficiently"""
        x, y = position
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        
        # Draw background rectangle
        cv2.rectangle(image, (x, y - text_h - 6), (x + text_w + 6, y), bg_color, -1)
        
        # Draw text
        cv2.putText(image, text, (x + 3, y - 3), self.font, self.font_scale, 
                   self.colors['text'], self.thickness, lineType=cv2.LINE_AA)

    def draw_vehicle_detection(self, image, vehicle_box, track_id, class_name, conf):
        """Draw vehicle detection box and label"""
        x1, y1, x2, y2 = map(int, vehicle_box.tolist())
        
        # Draw dashed box
        box_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        draw_dashed_polygon(image, box_corners, color=self.colors['vehicle_box'], 
                          thickness=1, dash_length=5)
        
        # Draw corner points
        for pt in box_corners:
            draw_circle(image, pt, color=self.colors['vehicle_box'], radius=3)
        
        # Draw vehicle label
        veh_label = f"ID {track_id} {class_name.capitalize()} {conf:.2f}"
        self.draw_text_with_background(image, veh_label, (x1, y1), self.colors['text_bg'])
        
        return x1, y1, x2, y2

    def draw_signal_result(self, image, signal, conf, x1, y2):
        """Draw signal recognition result"""
        signal_label = f"{signal} {conf:.2f}"
        
        # Choose background color based on signal type
        bg_color = self.colors.get(signal, self.colors['no_signal'])
        
        # Draw signal label
        (text_w, text_h), _ = cv2.getTextSize(signal_label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(image, (x1, y2), (x1 + text_w + 6, y2 + text_h + 6), bg_color, -1)
        cv2.putText(image, signal_label, (x1 + 3, y2 + text_h + 2), 
                   self.font, self.font_scale, self.colors['text'], 
                   self.thickness, lineType=cv2.LINE_AA)

    def process_vehicle_tracking(self, frame, vehicle_box, track_id):
        """Process vehicle tracking and signal recognition"""
        x1, y1, x2, y2 = map(int, vehicle_box.tolist())
        bbox = [x1, y1, x2, y2]
        
        # Update last seen frame for this track
        self.last_seen_frame[track_id] = self.frame_count
        
        # Check if vehicle is inside polygon and it's time to process
        if self.frame_count % self.interval == 0:
            self.vehicles[track_id]['under_line'] = is_below_line_check(bbox, self.line_check)
            self.vehicles[track_id]['inside_polygon'] = is_inside_polygon(bbox, self.polygon)
            if self.vehicles[track_id]['inside_polygon'] and self.vehicles[track_id]['under_line']:
                # Crop vehicle image
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size > 0:
                    self.vehicles[track_id]['frames'].append(Image.fromarray(vehicle_crop[:,:,::-1]))
                    self.vehicles[track_id]['if_track'] = True
            elif self.vehicles[track_id]['inside_polygon'] and self.vehicles[track_id]['if_track']:
                # Vehicle is inside polygon but not under line, still crop
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size > 0:
                        self.vehicles[track_id]['frames'].append(Image.fromarray(vehicle_crop[:,:,::-1]))
            frames_count = len(self.vehicles[track_id]['frames'])
            
            # Remove oldest frame if we exceed max frames
            if frames_count > self.MAX_FRAMES_PER_OBJECT:
                self.vehicles[track_id]['frames'].popleft()
            vehicle_outside = not is_inside_polygon(bbox, self.polygon)
            
            if (frames_count == self.MAX_FRAMES_PER_OBJECT or 
                (vehicle_outside and frames_count >= self.MIN_FRAMES_PER_OBJECT)):
                
                # Predict signal
                signal, conf = self.signal_recognitor.predict(self.vehicles[track_id]['frames'])
                self.signal_text[track_id] = (signal, conf)
                # self.vehicles[track_id] = []  # Clear frames
                        
            # Clean up vehicles that left polygon without enough frames
            elif vehicle_outside:
                if track_id in self.vehicles:
                    del self.vehicles[track_id]
                if track_id in self.last_seen_frame:
                    del self.last_seen_frame[track_id]

    def process_frame(self, frame):
        """Process a single frame"""
        image = frame.copy()
        
        # Draw polygon
        draw_dashed_polygon(image, self.polygon, color=self.colors['polygon'], 
                          thickness=2, dash_length=20)
        
        draw_line_check(image, self.line_check, color=self.colors['line_check'], 
                        thickness=2, dash_length=20)
        
        # Vehicle detection and tracking
        vehicle_results = self.vehicle_detector.track(image, persist=True, verbose=False)[0]
        
        if vehicle_results.boxes and vehicle_results.boxes.is_track:
            vehicle_boxes = vehicle_results.boxes.xyxy.cpu()
            confs = vehicle_results.boxes.conf.cpu().tolist()
            class_ids = vehicle_results.boxes.cls.int().cpu().tolist()
            track_ids = vehicle_results.boxes.id.int().cpu().tolist()
            
            for vehicle_box, conf, class_id, track_id in zip(vehicle_boxes, confs, class_ids, track_ids):
                # Skip low confidence detections
                if conf < self.CONF_THRESHOLD:
                    continue
                
                class_name = vehicle_results.names[class_id]
                
                # Draw vehicle detection
                x1, y1, x2, y2 = self.draw_vehicle_detection(image, vehicle_box, track_id, class_name, conf)
                
                # Process tracking and signal recognition
                self.process_vehicle_tracking(frame, vehicle_box, track_id)
                
                # Draw signal result if available
                if track_id in self.signal_text:
                    signal, signal_conf = self.signal_text[track_id]
                    self.draw_signal_result(image, signal, signal_conf, x1, y2)
        
        return image

    def cleanup_old_tracks(self):
        """Remove old tracks to prevent memory leaks"""
        current_frame = self.frame_count
        
        # Clean up tracks that haven't been updated recently
        tracks_to_remove = []
        
        for track_id in list(self.vehicles.keys()):
            # If track has been inactive for too long (no new frames added)
            if len(self.vehicles[track_id]['frames']) == 0:
                # Check if we have a last seen frame for this track
                if hasattr(self, 'last_seen_frame'):
                    if current_frame - self.last_seen_frame.get(track_id, 0) > self.fps * 2:  # 2 seconds
                        tracks_to_remove.append(track_id)
                else:
                    # Initialize last_seen_frame tracking
                    self.last_seen_frame = {}
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            if track_id in self.vehicles:
                del self.vehicles[track_id]
            if track_id in self.signal_text:
                del self.signal_text[track_id]
            if hasattr(self, 'last_seen_frame') and track_id in self.last_seen_frame:
                del self.last_seen_frame[track_id]
        
        # Limit maximum number of tracks to prevent memory overflow
        MAX_TRACKS = 100
        if len(self.vehicles) > MAX_TRACKS:
            # Remove oldest tracks (those with smallest track_id)
            oldest_tracks = sorted(self.vehicles.keys())[:len(self.vehicles) - MAX_TRACKS]
            for track_id in oldest_tracks:
                if track_id in self.vehicles:
                    del self.vehicles[track_id]
                if track_id in self.signal_text:
                    del self.signal_text[track_id]
                if hasattr(self, 'last_seen_frame') and track_id in self.last_seen_frame:
                    del self.last_seen_frame[track_id]

    def run(self, input_path, output_path):
        """Main processing loop"""
        self.setup_video(input_path, output_path)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write to output
                self.out.write(processed_frame)
                
                # Display (optional - can be disabled for better performance)
                # cv2.namedWindow("Signal Recognition", cv2.WINDOW_NORMAL)
                # cv2.imshow("Signal Recognition", processed_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                self.frame_count += 1
                
                # Periodic cleanup (every 5 seconds)
                if self.frame_count % (self.fps * 5) == 0:
                    self.cleanup_old_tracks()
                
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        # cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    config = {
        'conf_threshold': 0.8,
        'skip_time': 0.2,
        'min_frames': 5,
        'max_frames': 15,
        'vehicle_model_path': './weights/vehicle_detection_demo.pt',
        'signal_model_config': './crnn/config.yaml',
        'polygon_path': './samples/demo.txt',
        'line_check_path': './samples/line_check.txt'
    }
    
    recognizer = VehicleSignalRecognizer(config)
    recognizer.run('./samples/demo.mp4', './results/output.mp4')
    print("Processing completed!!!")