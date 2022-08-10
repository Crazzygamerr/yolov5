from msilib.schema import File
import os
import io
import json
import random
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

def plot_bounding_box(image, annotation_list):
		annotations = np.array(annotation_list)
		w, h = image.size
		
		plotted_image = ImageDraw.Draw(image)

		transformed_annotations = np.copy(annotations)
		transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
		transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
		
		transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
		transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
		transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
		transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
		
		for ann in transformed_annotations:
				obj_cls, x0, y0, x1, y1 = ann
				plotted_image.rectangle(((x0,y0), (x1,y1)))
				
				plotted_image.text((x0, y0 - 10), str(int(obj_cls)))
		
		plt.imshow(np.array(image))
		plt.show()

# Display a image with its annotations
def showAnnotation():
	annotations = [os.path.join('archive\\train\\labels', x) for x in os.listdir('archive\\train\\labels') if x.endswith('.txt')]

	random.seed(42)
	# annotation_file = random.choice(annotations)
	# annotation_file from bbigpwjfll_jpg.rf.0024120050110d4e2db1eec5fc8233de.txt
	annotation_file = os.path.join('archive\\train\\labels', 'bbigpwjfll_jpg.rf.0024120050110d4e2db1eec5fc8233de.txt')

	with open(annotation_file, "r") as file:
			annotation_list = file.read().split("\n")[:-1]
			annotation_list = [x.split(" ") for x in annotation_list]
			annotation_list = [[float(y) for y in x ] for x in annotation_list]

	image_file = annotation_file.replace("labels", "images").replace("txt", "jpg")
	assert os.path.exists(image_file)
	image = Image.open(image_file)
	plot_bounding_box(image, annotation_list)

def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image

# Sample function used run the model
def runModel():
	model = torch.hub.load('./', 'custom', path='./runs/train/yolo_test2/weights/best.pt', source='local')  # local repo
	model.conf = 0.25
	# image_file = os.path.join('modified_annot\\images', 'bwbjouzvzl_jpg.rf.de2ef467471546c6f59e31a9499a3424.jpg')
	image_files = [os.path.join('modified_annot\\images', x) for x in os.listdir('modified_annot\\images') if x.endswith('.jpg')]
	# load local file
	# image_file = File(os.path.join('archive\\test\\images', 'bwbjouzvzl_jpg.rf.de2ef467471546c6f59e31a9499a3424.jpg'))
	# input_image = get_image_from_bytes(open(image_file, "rb").read(), max_size=640)
	
	for index, image in enumerate(image_files):
		# if index > 10:
		# 	break
		results = model(image)
		# detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
		# detect_res = json.loads(detect_res)
		print("----")
		print(results)
		detect_res = results.pandas().xyxy[0].to_json(orient="records")
		detect_res = json.loads(detect_res)
		print(detect_res)
		results.show()

#  POC function to check image cropping
def cropImage():
	image_file = os.path.join('archive\\test\\images', 'bbigpwjfll_jpg.rf.340405a2d230dbba32cbf506cd9aa7e4.txt')
	input_image = Image.open(image_file)
	cv2.imshow("image", np.array(input_image))
	cv2.waitKey(0)
	input_image = input_image.crop((0, 0, 100, 100))
	cv2.imshow("image", np.array(input_image))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Checks all the annotation files for class 7
# Becase it was annotated with the label 66 in the old dataset and those images had to be removed
def checkLabels():
	# get all filenames from ./train/labels
	labels = [os.path.join('archive\\train\\labels', x) for x in os.listdir('archive\\train\\labels') if x.endswith('.txt')]
	labels += [os.path.join('archive\\test\\labels', x) for x in os.listdir('archive\\test\\labels') if x.endswith('.txt')]
	# labels += [os.path.join('archive\\valid\\labels', x) for x in os.listdir('archive\\valid\\labels') if x.endswith('.txt')]
	for label in labels:
		with open(label, "r") as file:
			for line in file.read().split("\n")[:-1]:
				# print(line)
				if line.startswith("7 "):
					print(label)
					# print("----------" + line)
			# annotation_list = file.read().split("\n")[:-1]
			# annotation_list = [x.split(" ") for x in annotation_list]
			# annotation_list = [[float(y) for y in x ] for x in annotation_list]
			# for ann in annotation_list:
			# 	if ann[0] == 66:
			# 		print(label)
			# 		break

# The new annotated dataset had '.' label for class 0, but it was needed for class 7
# This function moves all the labels with '.' to class 7 and moves the rest
# Done for all the annotation files
def modifyAnnotations():
	labels = [os.path.join('modified_annot\\labels', x) for x in os.listdir('modified_annot\\labels') if x.endswith('.txt')]
	for label in labels:
		with open(label, "r") as file:
			lines = file.read().split("\n")
			for index, line in enumerate(lines):
				if line == "":
					continue
				cname = int(line.split(" ")[0])	
				if cname == 0:
					lines[index] = "7 " + line[2:]
				elif cname <= 7 & cname >= 1:
					lines[index] = str(cname-1) + " " + line[2:]
			with open(label, "w") as file:
				file.write("\n".join(lines))
					
if __name__ == '__main__':
	# showAnnotation()
	# runModel()
	# cropImage()
	# checkLabels()
	# modifyAnnotations()
