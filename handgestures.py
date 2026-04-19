# Helper functions:

def is_peace_sign(hand_landmarks):
    # Check if index and middle fingers are extended (tip is above PIP joint)
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y
    
    # Check if ring and pinky are folded (tip is below PIP joint)
    ring_down = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_down = hand_landmarks[20].y > hand_landmarks[18].y

    return index_up and middle_up and ring_down and pinky_down