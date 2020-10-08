import cv2 as cv

thickness = 2
lineType = cv.LINE_AA
color1 = (255, 0, 0)
color2 = (0, 255, 0)
color3 = (0, 0, 255)
color4 = (150, 255, 245)
color5 = (245, 255, 150)
ponit_color1 = (128, 0, 0)
ponit_color2 = (0, 128, 0)
ponit_color3 = (0, 0, 128)
ponit_color4 = (128, 0, 128)

point_size = 4

color = [color1, color2, color3, color4, color5]
point_color = [ponit_color1, ponit_color2, ponit_color3, ponit_color4]
for i in range(720):
	img = cv.imread(f"D:\\ltt\\images\\{i + 1}.tif")  # f型字符串，计算{i+1}
	label = open(f"D:\\ltt\\labels\\{i + 1}.txt")
	file = label.read()
	content = file.replace("\n", " ")
	strr = content.split(" ")
	content_list = [int(i) for i in strr if i != ""]  # 将label数据转成list，值为int型
	# content_list = [int(i) for i in content_list]
	# print(content_list)

	content_length = 9  # 单条数据的长度
	length = len(content_list) // 9
	# print(length)
	for j in range(length):
		point1 = (content_list[j * content_length + 1], content_list[j * content_length + 2])
		point2 = (content_list[j * content_length + 3], content_list[j * content_length + 4])
		point3 = (content_list[j * content_length + 5], content_list[j * content_length + 6])
		point4 = (content_list[j * content_length + 7], content_list[j * content_length + 8])

		ship = content_list[j * content_length] - 1
		cv.line(img, point1, point2, color[ship], thickness, lineType)
		cv.line(img, point2, point3, color[ship], thickness, lineType)
		cv.line(img, point3, point4, color[ship], thickness, lineType)
		cv.line(img, point4, point1, color[ship], thickness, lineType)
		cv.circle(img, point1, point_size, point_color[0], -1)
		cv.circle(img, point2, point_size, point_color[1], -1)
		cv.circle(img, point3, point_size, point_color[2], -1)
		cv.circle(img, point4, point_size, point_color[3], -1)
	# cv.namedWindow("image")
	# cv.imshow('image', img)
	# cv.waitKey(10000)
	cv.imwrite(f"D:\\ltt\\test\\{i + 1}.tif", img)
	label.close()
