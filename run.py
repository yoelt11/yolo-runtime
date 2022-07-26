import cv2
import numpy as np
import onnxruntime as ort 
import onnx
import torch


class onnx_run():
	def __init__(self, model_path):
		self.sess = ort.InferenceSession(model_path)
		self.input_name = self.sess.get_inputs()[0].name
		self.label_name = self.sess.get_outputs()[0].name 


	def infer(self, x):
		outputs = self.sess.run([self.label_name], {self.input_name: x.astype(np.float32)})[0]
		return outputs

def letterbox(im, new_shape=(384, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
	# Resize and pad image while meeting stride-multiple constraints
	shape = im.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better val mAP)
		r = min(r, 1.0)

	# Compute padding
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
		top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
		left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
		im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

	return im, r, (dw, dh)

def image_preprocess(img, w, h):
	img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
	image = img.copy()
	image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	image, ratio, dwdh = letterbox(image, auto=False)
	image = image.transpose((2, 0, 1))
	image = np.expand_dims(image, 0)
	image = np.ascontiguousarray(image)

	x = image.astype(np.float32)
	x /= 255
	x.shape

	return  x, ratio, dwdh

def interpret_output(outputs, image, ratio, dwdh):

	if image.size != 0:
		h, w = .9473 ,1#image.shape[0], image.shape[1]
		threshold = .35
		for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
			if score >= threshold:
				box = np.array([x0,y0,x1,y1])
				box -= np.array(dwdh*2)
				box /= ratio
				box = box.round().astype(np.int32).tolist()
				cls_id = int(cls_id)
				name = names[cls_id]
				color = colors[name] #(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
				name += ' '+str(score)
				thickness = 2
				image = cv2.rectangle(image,box[:2],box[2:],color,thickness)
				image = cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

	return image, outputs


'''
Main Function
'''
if __name__ == '__main__':

	MODEL_PATH = './object-detection-models/YOLOv7/yolov7-tiny.onnx'
	inference_engine = onnx_run(MODEL_PATH)
	
	# x = np.random.rand(1,3,384,640) <== model input size
	width, height =  640, 384 # from model
	
	# Initialize video capture
	cap = cv2.VideoCapture(0)
	cap.set(4,360)
	cap.set(3,640)

	# classes from current model
	names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
	         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
	         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
	         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
	         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
	         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
	         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
	         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
	         'hair drier', 'toothbrush']
	# set specific color to each class
	colors = {name:[np.random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

	while True:
		# get image
		ret, image = cap.read()
		# converts into model's input form
		x, ratio, dwdh  = image_preprocess(image, width, height)
		# run inference
		outputs = inference_engine.infer(x)
		# interpret iference gets outputs and image with boxes
		image_out, outputs = interpret_output(outputs, image, ratio, dwdh)
		# resize image to original dimensions
		image_out = cv2.resize(image_out, (640, 360), interpolation=cv2.INTER_LINEAR)

		cv2.imshow("YOLOv7", image_out)

		if cv2.waitKey(1) == 27: 
			break  # esc to quit

