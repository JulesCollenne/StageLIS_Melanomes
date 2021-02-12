# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

image = img_as_float(io.imread("naevus.JPG"))

numSegments = 300

segments = slic(image, n_segments=numSegments, sigma=50)
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % numSegments)
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
plt.savefig("superpixel.jpg", bbox_inches='tight',
            pad_inches=0)
