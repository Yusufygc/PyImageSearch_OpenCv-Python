"""
Step #1: Detect keypoints (DoG, Harris, etc.) and extract local invariant descriptors (SIFT, SURF, etc.) from the two input images.
Step #2: Match the descriptors between the two images.
Step #3: Use the RANSAC algorithm to estimate a homography matrix using our matched feature vectors.
Step #4: Apply a warping transformation using the homography matrix obtained from Step #3.

Adım #1: Anahtar noktaları (DoG, Harris, vb.) tespit edin ve iki giriş görüntüsünden yerel değişmez tanımlayıcıları (SIFT, SURF, vb.) çıkaralım.
Adım #2: İki resim arasındaki tanımlayıcıları eşleştirelim.
Adım #3: Eşleştirilmiş öznitelik vektörlerimizi kullanarak bir homografi matrisini tahmin etmek için RANSAC algoritmasını kullanalım.
Adım #4: Adım #3'ten elde edilen homografi matrisini kullanarak bir çarpıtma dönüşümü uygulayalım.

"""
# import the necessary packages
import numpy as np
import imutils
import cv2

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3(or_better=True)
		
	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		"""
	    İsteğe bağlı olarak, özellikleri eşleştirirken David Lowe'un oran testi için kullanılan oran, RANSAC algoritmasının izin verdiği maksimum piksel "hareket alanı" olan reprojThresh ve son olarak bir boolean olan showMatches'ı sağlayabiliriz. anahtar nokta eşleşmelerinin görselleştirilmesinin gerekip gerekmediğini belirtmek için kullanılır.
	    """
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images # görüntülerin soldan sağa sıralanmasını bekliyoruz.
		# Bu sırayla sağlanmazsa kodumuz çalışmaya devam eder, 
        # ancak çıktı panoramamaız yalnızca bir görüntü içerir.
		(kpsA, featuresA) = self.detectAndDescribe(imageA) # görüntü A'dan anahtar noktaları ve özellikleri çıkarırız.
		(kpsB, featuresB) = self.detectAndDescribe(imageB) # görüntü B'den anahtar noktaları ve özellikleri çıkarırız.
		
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh) # iki görüntü arasındaki özellikleri eşleştiririz.
		
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None: # eşleşme yoksa, o zaman bir panoramik oluşturmak için yeterli eşleşen anahtar nokta yoktur.
			return None
		
		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M # eşleşmeler, H ve durum değişkenlerini çıkarırız.
		# H, görüntü A'yı görüntü B'ye kaydırmak için kullanılan homografi matrisidir.
        # status, eşleşmelerin başarılı olup olmadığını belirtir.
		""" 
		M değeri None olmadığı sürece, 47. satırda demetin içeriğini açarız, bize bir anahtar nokta eşleme listesi, RANSAC algoritmasından türetilmiş homografi matrisi H ve son olarak, RANSAC kullanılarak başarılı bir şekilde mekânsal olarak doğrulanan anahtar noktalarının hangileri olduğunu gösteren bir dizin listesi olan "status"ı elde ederiz.
		"""

		"""
		Homografi matrisi H'ye sahip olduğumuza göre, şimdi iki görüntüyü bir araya getirmeye hazırız. İlk olarak, cv2.warpPerspective işlevini üç argümanla çağırıyoruz: dönüştürmek istediğimiz görüntü (bu durumda sağdaki görüntü), 3 x 3 dönüşüm matrisi (H) ve son olarak çıktı görüntüsünün şekli. Çıktı görüntüsünün şeklini, her iki görüntünün genişliklerinin toplamını alarak ve ardından ikinci görüntünün yüksekliğini kullanarak elde ederiz.
		"""
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0])) 
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB 
		
		# check to see if the keypoint matches should be visualized
		if showMatches: # anahtar nokta eşleşmelerinin görselleştirilmesi gerekip gerekmediğini kontrol ederiz.
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)
			
			# return a tuple of the stitched image and the
			# visualization
			return (result, vis) # dikilmiş(stitch) görüntü ve görselleştirme demetini döndürürüz.
		
		# return the stitched image
		return result