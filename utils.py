import math

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def get_class_color(cls):
    """
    Simple function that adds fixed color depending on the class
    """
    if cls == 'car':
        color = (204, 51, 0)
    elif cls == 'truck':
        color = (22,82,17)
    elif cls == 'motorbike':
        color = (255, 0, 85)
    else:
        color = [int((p * (2 ** 2 - 14 + 1)) % 255) for p in palette]
    return tuple(color)

def estimatedSpeed(location1, location2):
    #Euclidean distance formula
    d_pixel = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # setting the pixels per meter 
    ppm = 4 # This value could me made dynamic depending on how close the object is from the camera
    d_meters = d_pixel/ppm
    time_constant = 15*3.6

    speed = (d_meters * time_constant)/100
    return int(speed)