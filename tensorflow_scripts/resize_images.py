from PIL import Image
import numpy as np
import os

root = "ExtendedYaleB/"
root2 = "ExtendedYaleB_resized/"

image_width = 100
image_height = 100

pixel_depth = 255.0

for folder in os.listdir(root):
	image_files = os.listdir(root + folder + "/")

	print("Redimensionando diretorio " + folder)

	for image in image_files:
		image_file = root + folder + "/" + image
		path = root2 + folder

		if(not os.path.isdir(path)):
			os.makedirs(path)

		try:
			img = Image.open(image_file)
			img = img.resize((image_width, image_height), Image.ANTIALIAS)
			img.save(root2 + folder + "/" + image)
		except IOError as e:
			print("Impossivel ler " + image_file)

'''
img = Image.open(img_path)

print img.size[0], img.size[1]
img.show()

img = img.resize((img.size[0]/18, img.size[1]/18), Image.ANTIALIAS)

print img.size[0], img.size[1]
img.show()

img.save("resized.jpg")'''