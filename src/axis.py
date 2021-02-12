
# Takes a mask and return the number of white pixels
def compute_area(img):
    return len([pixel for row in img for pixel in row if pixel != 0])


# Using eigenvectors of covariance matrix
def get_axis(img):
    pass
