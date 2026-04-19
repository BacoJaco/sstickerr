import cv2
import os

stickers = {}

def load_all_images():    
    image_paths = {
        "peace": "./GFProvidedHamsters/Peace.png",
    }

    for name, path in image_paths.items():
        if os.path.exists(path):
            # Load with alpha channel
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            
            if img is not None:
                # Resize them uniforml
                stickers[name] = cv2.resize(img, (150, 150))
            else:
                print(f"Error: OpenCV could not read {path}.")
        else:
            print(f"Error: File not found at {path}.")

def overlay_transparent(background, overlay, x, y):
    # Overlays a transparent PNG onto a background image at position (x, y)
    h, w = overlay.shape[0], overlay.shape[1]
    
    # Check if the overlay goes out of bounds, and clip it if necessary
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    
    y1o, y2o = max(0, -y), min(h, background.shape[0] - y)
    x1o, x2o = max(0, -x), min(w, background.shape[1] - x)
    
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return background

    overlay_crop = overlay[y1o:y2o, x1o:x2o]
    bg_crop = background[y1:y2, x1:x2]

    # If the overlay has a transparent bg
    if overlay.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            bg_crop[:, :, c] = (alpha * overlay_crop[:, :, c] + alpha_inv * bg_crop[:, :, c])
    else:
        background[y1:y2, x1:x2] = overlay_crop

    return background