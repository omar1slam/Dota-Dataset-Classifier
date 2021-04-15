from imageio import imread
import os
from matplotlib import pyplot
import numpy as np
import cv2
from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


def ret_2nd_ele(tuple_1):
    return tuple_1[1]

class DotaDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir,isTrain = True):
        # define classes
        self.add_class("dataset", 1, "plane")
        self.add_class("dataset", 2, "ship")
        self.add_class("dataset", 3, "storage-tank")
        self.add_class("dataset", 4, "basketball-court")
        self.add_class("dataset", 5, "baseball-diamond")
        self.add_class("dataset", 6, "tennis-court")
        self.add_class("dataset", 7, "basketball-court")
        self.add_class("dataset", 8, "ground-track-field")
        self.add_class("dataset", 9, "harbor")
        self.add_class("dataset", 10, "bridge")
        self.add_class("dataset", 11, "large-vehicle")
        self.add_class("dataset", 12, "small-vehicle")
        self.add_class("dataset", 13, "helicopter")
        self.add_class("dataset", 14, "roundabout")
        self.add_class("dataset", 15, "soccer-ball-field")
        self.add_class("dataset", 16, "swimming-pool")


        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/labels/'

        # find all images
        for filename in os.listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            img_check = image_id[1:]
            if isTrain and int(img_check) <= 350:
                continue
            # skip all images before 150 if we are building the test/val set
            if not isTrain and int(img_check) > 350:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.txt'

            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self , filename):
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
            del lines[0]
            del lines[0]
            for i in range(len(lines)):
                lines[i] = lines[i].split()

        pts = []
        for l in lines:
            x = [int(float(l[i])) for i in range(0, 7, 2)]
            y = [int(float(l[j])) for j in range(1, 8, 2)]
            cords = list(zip(x, y))
            cords.append(l[-2])
            pts.append(cords)
            # pts
            # pts = np.array(pts, np.int32)
            # pts = pts.reshape((-1, 1, 2))

        return pts

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        a = cv2.imread(info['path'])
        h ,w = a.shape[0],a.shape[1]
        boxes = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()

        for i in range(len(boxes)):
            # for j in range(len(boxes[i])-1):
            maxmin = boxes[i][:-1]
            xmin = min(maxmin)
            xmax = max(maxmin)
            ymin = min(maxmin, key=ret_2nd_ele)
            ymax = max(maxmin, key=ret_2nd_ele)
                
            masks[ymin[1]:ymax[1],xmin[0]:xmax[0],i] = 1
            class_ids.append(self.class_names.index(boxes[i][-1]))

        return masks, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# define a configuration for the model
class DotaConfig(Config):
    # Give the configuration a recognizable name
    NAME = "Dota_cfg"
    # Number of classes 
    NUM_CLASSES = 17
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGES_PER_GPU = 1
    LEARNING_RATE=0.001
    WEIGHT_DECAY = 0.0001
    TRAIN_ROIS_PER_IMAGE = 500
    MAX_GT_INSTANCES = 250
    
    
# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "Dota_cfg"
    # number of classes 
    NUM_CLASSES = 17
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    


def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = np.expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = np.mean(APs)
	return mAP
 
# print(extract_boxes('E:\Stuff\Work\Wekala\Datasets\DOTA\labels/P0000.txt'))
prev_epoch = r'DOTA_test\dota_cfg20201230T1045/' + r'mask_rcnn_dota_cfg_0005.h5'

train_set = DotaDataset()
train_set.load_dataset(r'DOTA/',isTrain=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
test_set = DotaDataset()
test_set.load_dataset(r'DOTA/',isTrain=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


 
# prepare config
config = DotaConfig()
config.display()

# # define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# create config
# cfg = PredictionConfig()
# cfg.display()
# # define the model
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load weights 
model.load_weights(prev_epoch, by_name=True,exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads') ,exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
# evaluate model on training dataset
# train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# # evaluate model on test dataset
# test_mAP = evaluate_model(test_set, model, cfg)
# print("Test mAP: %.3f" % test_mAP)

