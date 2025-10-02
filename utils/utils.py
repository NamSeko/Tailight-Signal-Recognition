import cv2
import numpy as np


def load_polygon(txt_path):
    with open(txt_path, 'r') as f:
        polygon = [tuple(map(int, line.strip().split(','))) for line in f]
    return np.array(polygon, dtype=np.int32)

def is_inside_polygon(box, polygon):
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, y2  # bottom-center
    return cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0

# Convert line_check to functional form y=f(x)
def interpolate_line_check(x_center, line_check):
    for i in range(len(line_check) - 1):
        x1, y1 = line_check[i]
        x2, y2 = line_check[i + 1]
        if x1 <= x_center <= x2 or x2 <= x_center <= x1:
            t = (x_center - x1) / (x2 - x1 + 1e-6)
            y = y1 + t * (y2 - y1)
            return int(y)
    return None

# Check if the bottom-center of the box is below the line_check
def is_below_line_check(box, line_check):
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, y2  # bottom-center
    
    # Lấy y trung bình của line_check
    line_y = interpolate_line_check(cx, line_check)
    if line_y is None:
        return False
    return cy > line_y


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=5):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    for i in np.arange(0, dist, dash_length * 2):
        start_ratio = i / dist
        end_ratio = min(i + dash_length, dist) / dist
        x1 = int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio)
        y1 = int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
        x2 = int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio)
        y2 = int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

def draw_dashed_polygon(img, points, color, thickness=1, dash_length=5):
    n = len(points)
    for i in range(n):
        pt1 = tuple(points[i])
        pt2 = tuple(points[(i + 1) % n])
        draw_dashed_line(img, pt1, pt2, color, thickness, dash_length)
        
def draw_line_check(img, points, color, thickness=1, dash_length=5):
    n = len(points)
    for i in range(n-1):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])
        draw_dashed_line(img, pt1, pt2, color, thickness, dash_length)

def draw_circle(img, center, color, radius=3):
    cv2.circle(img, tuple(int(x) for x in center), radius, color, -1, lineType=cv2.LINE_AA)