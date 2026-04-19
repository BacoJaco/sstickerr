import math

# Helper functions:

def is_peace_sign(hand_landmarks):
    # Check if index and middle fingers are extended (tip is above PIP joint)
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y
    
    # Check if ring and pinky are folded (tip is below PIP joint)
    ring_down = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_down = hand_landmarks[20].y > hand_landmarks[18].y

    return index_up and middle_up and ring_down and pinky_down

def is_gun(hand_landmarks):    
    # Index finger extended
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    
    # Middle, Ring, and Pinky fingers folded down
    middle_down = hand_landmarks[12].y > hand_landmarks[10].y
    ring_down = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_down = hand_landmarks[20].y > hand_landmarks[18].y
    
    # Thumb pointing up
    thumb_up = hand_landmarks[4].y < hand_landmarks[3].y
    
    return index_up and middle_down and ring_down and pinky_down and thumb_up

# Calculate the Euclidean distance between two landmarks
def calculate_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

def is_heart(hands):
    # A heart requires exactly two hands
    if len(hands) != 2:
        return False
    
    hand1 = hands[0]
    hand2 = hands[1]
    
    # Thumbs touching
    thumb_distance = calculate_distance(hand1[4], hand2[4])
    thumbs_touching = thumb_distance < 0.05  # 0.05 is roughly 5% of the screen
    
    # Indexes touching
    index_distance = calculate_distance(hand1[8], hand2[8])
    index_touching = index_distance < 0.05

    # Thumbs below indexes
    thumbs_are_bottom = (hand1[4].y > hand1[8].y) and (hand2[4].y > hand2[8].y)
    
    return thumbs_touching and index_touching and thumbs_are_bottom

def is_begging(hands):
    hand1 = hands[0]
    hand2 = hands[1]

    # Pinkies close
    pinky_base_dist = calculate_distance(hand1[17], hand2[17])
    pinkies_touching = pinky_base_dist < 0.15
    
    # The indexes should be further apart to form the "bowl"
    index_tip_dist = calculate_distance(hand1[8], hand2[8])
    is_bowl_shape = index_tip_dist > 0.20
    
    # Hands upwards
    upwards = (hand1[12].y < hand1[0].y) and (hand2[12].y < hand2[0].y)

    return pinkies_touching and is_bowl_shape and upwards

# Returns the name of the detected gesture
def detect_gesture(hand_landmarks):
    if not hand_landmarks:
        return None

    # Check for two-hand gestures first
    if len(hand_landmarks) == 2:
        if is_heart(hand_landmarks):
            return "heart"
        if is_begging(hand_landmarks):
            return "beg"
        
    # If no two-hand gesture is found, check each hand for single-hand gestures
    for hand in hand_landmarks:
        if is_peace_sign(hand):
            return "peace"

        if is_gun(hand):
            return "gun"

    return None