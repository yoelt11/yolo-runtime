import cv2
import numpy as np
import torch
from torchvision import transforms
import sys
sys.path.insert(0,'./')
from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.layers import paste_masks_in_image

class Yolov7():
	def __init__(self, model_path):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		weights = torch.load(model_path)
		self.interpreter = weights['model']
		self.interpreter = self.interpreter.half().to(self.device)
		_ = self.interpreter.eval()

	def infer(self, x):
		outputs = self.interpreter(x.to(torch.float16))
		return outputs

def image_preprocess(image):
    in_image = letterbox(image, 640, stride=64, auto=True)[0]
    in_image = transforms.ToTensor()(in_image)
    in_image = torch.tensor(np.array([in_image.numpy()]))
    in_image = in_image.half().to(inference_engine.device)
    image = torch.tensor(image).unsqueeze(0).numpy()
    return in_image, image

def interpret_output(output):
    hyp = {'mask_resolution':56, 'attn_resolution':14 , 'num_base':5}
    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = x.shape
    names = inference_engine.interpreter.names
    pooler_scale = inference_engine.interpreter.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
    pred, pred_masks, base = output[0], output_mask[0], bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = paste_masks_in_image(original_pred_masks, bboxes, (height, width), threshold=0.5)

    # bring tensor to cpu and conver to numpy
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int64)
    
    results = {'cls': pred_cls, 'boxes': nbboxes, 'masks': pred_masks_np, 'score': pred_conf}

    return results

def plot2image(results, x, colors):
    num_obj  = results['boxes'].shape[0]
    bbox = results['boxes']
    cls = results['cls']
    masks = results['masks']
    scores = results['score']
    c = list(colors.values())
    k = list(colors.keys())
    nimg =(x[0].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
    for i in range(num_obj):
        if scores[i] > .50:
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            nimg = cv2.rectangle(nimg, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), c[int(cls[i])], 2) 
            nimg = cv2.putText(nimg, k[int(cls[i])], (bbox[i][0], bbox[i][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (60, 100, 255), 2)
            nimg = cv2.putText(nimg, str(scores[i])[:3], (bbox[i][0], bbox[i][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 200, 20), 2)
            nimg[masks[i]] = nimg[masks[i]] * 0.5 + np.array(c[int(cls[i])], dtype=np.uint8) * 0.5

    return nimg

#TODO overlapping when person touches object

'''
Main Function
'''
if __name__ == '__main__':

	MODEL_PATH = '/home/etorres/Documents/in-work/computer-vision-models/object-detection-models/YOLOv7/yolov7-mask2.pt'
	inference_engine = Yolov7(MODEL_PATH)
	
	width, height =  640, 384 # from model
	
	# Initialize video capture
	cap = cv2.VideoCapture(0)
	cap.set(4,360)
	cap.set(3,640)

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
		x, image = image_preprocess(image)
		# run inference
		outputs = inference_engine.infer(x)
		# interpret iference gets outputs and image with boxes
		outputs = interpret_output(outputs)
        # plot to image
		image_out = plot2image(outputs, x, colors)
		# resize image to original dimensions
		image_out = cv2.resize(image_out, (640, 360), interpolation=cv2.INTER_LINEAR)
		print(image_out.shape)
		cv2.imshow("YOLOv7", image_out)

		if cv2.waitKey(1) == 27: 
			break  # esc to quit

