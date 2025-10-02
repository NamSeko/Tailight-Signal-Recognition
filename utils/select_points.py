import cv2

# Danh sách lưu các điểm được chọn
points = []

# Hàm xử lý sự kiện click chuột
def click_event(event, x, y, flags, param):
    global points, img

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 2)
        cv2.imshow('Image', img)

# Đọc ảnh
img = cv2.imread('../samples/demo.jpg')

# Tạo cửa sổ với chế độ cho phép resize
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

# Hiển thị ảnh và gán hàm click
cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)

# Chờ người dùng nhấn phím
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu điểm vào file
with open('../samples/demo.txt', 'w') as f:
    for pt in points:
        f.write(f"{pt[0]},{pt[1]}\n")

print("Các điểm đã chọn:", points)
print("Đã lưu vào file points.txt")
