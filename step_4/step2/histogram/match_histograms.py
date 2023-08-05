"""Unutmayın, burada yaptığımız renk dağılımını,referanstan alıp kaynağa aktarmaktır."""
# import the necessary packages
from skimage import exposure
# Görüntü histogramlarını, kümülatif dağılım fonksiyonlarını hesaplamak ve histogram eşleştirmeyi uygulamak için scikit-image'ın pozlama kitaplığına ihtiyacımız var.
import matplotlib.pyplot as plt
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
	help="path to the input source image")
ap.add_argument("-r", "--reference", required=True,
	help="path to the input reference image")
args = vars(ap.parse_args())

# load the source and reference images
print("[INFO] loading source and reference images...")
src = cv2.imread(args["source"])
ref = cv2.imread(args["reference"])

# determine if we are performing multichannel histogram matching
# and then perform histogram matching itself
"""Öncelikle, çok kanallı histogram eşleştirme işlemini gerçekleştirip gerçekleştirmediğimizi belirleyelim. Daha sonra histogram eşitleme işlemini uygulayalım.Çok kanallı histogram eşleştirme işlemini gerçekleştirip gerçekleştirmediğimizi belirleyelim."""

print("[INFO] performing histogram matching...")
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=multi)

# show the output images
cv2.imshow("Source", src)
cv2.imshow("Reference", ref)
cv2.imshow("Matched", matched)
cv2.waitKey(0)

"""Bu noktada teknik olarak işimiz bitti, ancak histogram eşleştirmenin ne yaptığını tam olarak anlamak için src, ref ve eşleşen görüntülerin renk histogramlarını inceleyelim:"""

# construct a figure to display the histogram plots for each channel
# before and after histogram matching was applied
(fig, axs) =  plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

# loop over our source image, reference image, and output matched image
for (i, image) in enumerate((src, ref, matched)):
	# convert the image from BGR to RGB channel ordering
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# loop over the names of the channels in RGB order
	for (j, color) in enumerate(("red", "green", "blue")):
		# compute a histogram for the current channel and plot it
		(hist, bins) = exposure.histogram(image[..., j],
			source_range="dtype")
		axs[j, i].plot(bins, hist / hist.max())

		# compute the cumulative distribution function for the
		# current channel and plot it
		(cdf, bins) = exposure.cumulative_distribution(image[..., j])
		axs[j, i].plot(bins, cdf)

		# set the y-axis label of the current plot to be the name
		# of the current color channel
		# Mevcut çizimin y ekseni etiketini, mevcut renk kanalının adı olarak ayarlayın.
		axs[j, 0].set_ylabel(color)

"""
Line 34 creates a 3 x 3 figure to display the histograms of the Red, Green, and Blue channels for each of the src, ref, and matched images, respectively.

From there, Line 38 loops over each of our src, ref, and matched images. We then convert the current image from BGR to RGB channel ordering.

Next comes the actual plotting:

Lines 45 and 46 compute a histogram for the current channel of the current image
We then plot that histogram on Line 47

Similarly, Lines 51 and 52 compute the cumulative distribution function of the current channel and then plot it
Line 56 sets the y-axis label of the color
The final step is to display the plot
"""

# set the axes titles
axs[0, 0].set_title("Source")
axs[0, 1].set_title("Reference")
axs[0, 2].set_title("Matched")

# display the output plots
plt.tight_layout()
plt.show()













