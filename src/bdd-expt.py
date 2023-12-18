import os
import torchvision
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from helpers.engine import train_one_epoch, evaluate

import torch
from torch.utils import data
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
from helpers.coco_eval import CocoEvaluator
from helpers.coco_utils import get_coco_api_from_dataset

from helpers.utils import load_json

class_map = {
    "bus": 1,
    "car": 2,
    "motor": 3,
    "person": 4,
    "rider": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "bike": 8,
    "truck": 9,
}


class BDDDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = None
        self.prepare()

    def prepare(self):
        self.samples = []

        if self.train:
            annotations = load_json(
                os.path.join(self.root, "labels/bdd100k_labels_images_train.json")
            )
            annotations = annotations[:5000]
            image_dir = os.path.join(self.root, "images/100k/train")
        else:
            annotations = load_json(
                os.path.join(self.root, "labels/bdd100k_labels_images_val.json")
            )
            # annotations = annotations[6000:6500]
            image_dir = os.path.join(self.root, "images/100k/val")

        for idx, ann in enumerate(annotations):
            ## filter instances of "lane" and "drivable_area", because they have poly2d instead of box2d
            invalid_idxs = [
                i
                for i, x in enumerate(ann["labels"])
                if x["category"] in ["lane", "drivable area", "train"]
            ]
            if len(invalid_idxs) == len(ann["labels"]):
                continue

            ann["labels"] = [
                ann["labels"][i]
                for i in range(len(ann["labels"]))
                if i not in invalid_idxs
            ]

            target = {}
            target["boxes"] = [
                ann["labels"][i]["box2d"] for i in range(len(ann["labels"]))
            ]
            target["boxes"] = [
                [box["x1"], box["y1"], box["x2"], box["y2"]] for box in target["boxes"]
            ]
            target["labels"] = [
                class_map[ann["labels"][i]["category"]]
                for i in range(len(ann["labels"]))
            ]
            target["image_id"] = idx + 1
            target["area"] = [
                (box[3] - box[1]) * (box[2] - box[0]) for box in target["boxes"]
            ]
            target["iscrowd"] = [0 for _ in target["boxes"]]
            # no mask

            image_path = os.path.join(image_dir, ann["name"])

            if os.path.exists(image_path):
                self.samples.append((image_path, target))
            else:
                raise FileNotFoundError

    def __getitem__(self, index):
        image_path, annotation = self.samples[index]

        image = pil_loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        # image = tv_tensors.Image(image)
        target = {}
        target["boxes"] = (
            torch.tensor(annotation["boxes"], dtype=torch.float).clone().detach()
        )
        target["labels"] = (
            torch.tensor(annotation["labels"], dtype=torch.int64).clone().detach()
        )
        target["area"] = (
            torch.tensor(annotation["area"], dtype=torch.float).clone().detach()
        )
        target["iscrowd"] = (
            torch.tensor(annotation["iscrowd"], dtype=torch.int64).clone().detach()
        )
        target["image_id"] = annotation["image_id"]

        return image, target

    def __len__(self):
        return len(self.samples)


def custom_collate_fn(batch):
    images, annotations = zip(*batch)
    images = data.dataloader.default_collate(images)
    annotations = list(annotations)
    return images, annotations


# transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor()])
loader_train = data.DataLoader(
    BDDDataset("../data/bdd100k", transform=transform),
    batch_size=1,
    shuffle=True,
    num_workers=8,
    collate_fn=custom_collate_fn,
)

loader_val = data.DataLoader(
    BDDDataset("../data/bdd100k", transform=transform, train=False),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    collate_fn=custom_collate_fn,
)

## do a baseline model

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
)

num_classes = 9 + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


## num of params
print(
    f"train params / params  = {sum(p.numel() for p in model.parameters() if p.requires_grad)} / {sum(p.numel() for p in model.parameters())}"
)


params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = model.to(device)


# with open('bdd_coco_evaluator.pkl', 'rb') as f:
# coco = pickle.load(f)['coco']

iou_types = ["bbox"]
coco = get_coco_api_from_dataset(loader_val.dataset)
coco_evaluator = CocoEvaluator(coco, iou_types)

# with open('bdd_coco_evaluator.pkl', 'wb') as f:
# pickle.dump({"coco": coco}, f)

# let's train it just for 2 epochs
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_metrics = train_one_epoch(
        model, optimizer, loader_train, device, epoch, print_freq=10
    )
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    eval_metrics = evaluate(model, loader_val, device, coco_evaluator)
    break
