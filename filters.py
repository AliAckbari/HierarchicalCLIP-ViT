import cv2
def v1_filtering(image):
    filtered_channels = []
    for i in range(3): 
        channel = image[:, :, i]
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        filtered_channels.append(edges)
    
    filtered_image = np.stack(filtered_channels, axis=-1)
    
    return filtered_image

def v2_filtering(image):
    radius = 1
    n_points = 8 * radius

    image = cv2.bilateralFilter(image, 9, 50, 50)
    lbp_channels = []
    edge_channels = []

    for i in range(3):  # Assuming the input image is RGB
        channel = image[:, :, i]
        channel = cv2.GaussianBlur(channel, (5, 5), 0)
        lbp = local_binary_pattern(channel, n_points, radius, method="uniform")
        lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lbp_channels.append(lbp)
        edge_channels.append(sobel)
    lbp_image = np.stack(lbp_channels, axis=-1)
    edge_image = np.stack(edge_channels, axis=-1)
    combined_image = cv2.addWeighted(lbp_image, .5, edge_image, 1, 0.5)

    return combined_image




def v3_filtering(image):
    filtered_channels = []
    for i in range(3):  
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1] 
        filtered_channels.append(saturation)
    filtered_image = np.stack(filtered_channels, axis=-1)
    
    return filtered_image




if name == "__main__":
    #some random arrays for testing, 3 * 128 * 128 