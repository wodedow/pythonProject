import cv2 as cv

img = cv.imread("./flower01.jpg")

height = img.shape[0]
width = img.shape[1]
channel = img.shape[2]

for x in range(height):
	for y in range(width):
		for z in range(channel):
			pixel = img[x, y, z]
			img[x, y, z] = pixel // 5 * 5

cv.namedWindow("test")
cv.imshow("test", img)
cv.waitKey(20000)
cv.imwrite("./test.jpg", img)
