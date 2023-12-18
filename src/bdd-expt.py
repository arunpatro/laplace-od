import wandb
from datetime import datetime
import os
import torchvision
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from helpers.engine import train_one_epoch, evaluate, get_metrics

import torch
from torch.utils import data
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
from helpers.coco_eval import CocoEvaluator
from helpers.coco_utils import get_coco_api_from_dataset
from helpers.utils import load_json
from netcal.metrics import ECE

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
            annotations = annotations[:1000]
            image_dir = os.path.join(self.root, "images/100k/train")
        else:
            annotations = load_json(
                os.path.join(self.root, "labels/bdd100k_labels_images_val.json")
            )
            image_dir = os.path.join(self.root, "images/100k/val")

        for idx, ann in enumerate(annotations):
            ## filter instances of "lane" and "drivable_area", because they have poly2d instead of box2d
            invalid_idxs = [
                i
                for i, x in enumerate(ann["labels"])
                if x["category"] in ["lane", "drivable area", "train"]
            ]

            ## if all instances are invalid, skip this image
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


def get_dataloaders():
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

    return loader_train, loader_val


def get_model(model_name="fasterrcnn_mobilenet_v3_large_fpn"):
    ## do a baseline model
    if model_name != "fasterrcnn_mobilenet_v3_large_fpn":
        raise NotImplementedError

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    )

    num_classes = 9 + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    ## add dropout to model -> Wrong could be ReLU
    # # Add dropout to RPN Head
    # rpn_conv_layers = [nn.Dropout(0.1), model.rpn.head.conv[0], model.rpn.head.conv[1]]
    # model.rpn.head.conv = nn.Sequential(*rpn_conv_layers)

    # rpn_cls_layers = [nn.Dropout(0.1), model.rpn.head.cls_logits]
    # model.rpn.head.cls_logits = nn.Sequential(*rpn_cls_layers)

    # rpn_bbox_layers = [nn.Dropout(0.1), model.rpn.head.bbox_pred]
    # model.rpn.head.bbox_pred = nn.Sequential(*rpn_bbox_layers)

    # # Add dropout to ROI Heads
    # roi_box_head_layers = [nn.Dropout(0.1), model.roi_heads.box_head.fc6,
    #                     nn.Dropout(0.1), model.roi_heads.box_head.fc7]
    # model.roi_heads.box_head = nn.Sequential(*roi_box_head_layers)

    # roi_cls_layers = [nn.Dropout(0.1), model.roi_heads.box_predictor.cls_score]
    # model.roi_heads.box_predictor.cls_score = nn.Sequential(*roi_cls_layers)

    # roi_bbox_layers = [nn.Dropout(0.1), model.roi_heads.box_predictor.bbox_pred]
    # model.roi_heads.box_predictor.bbox_pred = nn.Sequential(*roi_bbox_layers)

    ## num of params
    print(
        f"train {sum(p.numel() for p in model.parameters() if p.requires_grad)} / {sum(p.numel() for p in model.parameters())}"
    )
    return model


if __name__ == "__main__":
    expt_name = "baseline"
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_dir = f"ckpts/{expt_name}/{time_stamp}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    run = wandb.init(
        project="bml-od",
        name=f"{expt_name}-{time_stamp}",
        config={
            "model": "fasterrcnn_mobilenet_v3_large_fpn",
            "dataset": "bdd100k",
            "data_subset": "5k",
            "model_subset": "all",
            "time_stamp": time_stamp,
        },
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loader_train, loader_val = get_dataloaders()
    model = get_model(model=run["model"])

    model = model.to(device)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_model = swa_model.to(device)
    swa_start = 2
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

    iou_types = ["bbox"]
    coco = get_coco_api_from_dataset(loader_val.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # with open('bdd_coco_evaluator.pkl', 'wb') as f:
    # pickle.dump({"coco": coco}, f)

    num_epochs = 5

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_metrics = train_one_epoch(
            model, optimizer, loader_train, device, epoch, print_freq=10, wandbrun=run
        )

        # update the learning rate
        lr_scheduler.step()
        if epoch > swa_start:
            swa_model.update_parameters(model)
        torch.optim.swa_utils.update_bn(loader_train, swa_model)
        # train_swa_metrics = evaluate_model(swa_model, trainloader, criterion, device)
        # val_swa_metrics = evaluate_model(swa_model, valloader, criterion, device)
        # test_swa_metrics = evaluate_model(swa_model, testloader, criterion, device)

        # evaluate on the test dataset
        eval_metrics = evaluate(model, loader_val, device, coco_evaluator, wandbrun=run)

        t_metrics = get_metrics(model, loader_train, device)
        v_metrics = get_metrics(model, loader_val, device)

        swa_t_metrics = get_metrics(swa_model, loader_train, device)
        swa_v_metrics = get_metrics(swa_model, loader_val, device)

        train_prop_ece = ECE(bins=10).measure(
            t_metrics["ece_data"]["prop"]["probs"].detach().cpu().numpy(),
            t_metrics["ece_data"]["prop"]["labels"].detach().cpu().numpy(),
        )

        train_det_ece = ECE(bins=10).measure(
            t_metrics["ece_data"]["det"]["probs"].detach().cpu().numpy(),
            t_metrics["ece_data"]["det"]["labels"].detach().cpu().numpy(),
        )
        val_prop_ece = ECE(bins=10).measure(
            v_metrics["ece_data"]["prop"]["probs"].detach().cpu().numpy(),
            v_metrics["ece_data"]["prop"]["labels"].detach().cpu().numpy(),
        )
        val_det_ece = ECE(bins=10).measure(
            v_metrics["ece_data"]["det"]["probs"].detach().cpu().numpy(),
            v_metrics["ece_data"]["det"]["labels"].detach().cpu().numpy(),
        )

        swa_train_prop_ece = ECE(bins=10).measure(
            swa_t_metrics["ece_data"]["prop"]["probs"].detach().cpu().numpy(),
            swa_t_metrics["ece_data"]["prop"]["labels"].detach().cpu().numpy(),
        )
        swa_train_det_ece = ECE(bins=10).measure(
            swa_t_metrics["ece_data"]["det"]["probs"].detach().cpu().numpy(),
            swa_t_metrics["ece_data"]["det"]["labels"].detach().cpu().numpy(),
        )
        swa_val_prop_ece = ECE(bins=10).measure(
            swa_v_metrics["ece_data"]["prop"]["probs"].detach().cpu().numpy(),
            swa_v_metrics["ece_data"]["prop"]["labels"].detach().cpu().numpy(),
        )
        swa_val_det_ece = ECE(bins=10).measure(
            swa_v_metrics["ece_data"]["det"]["probs"].detach().cpu().numpy(),
            swa_v_metrics["ece_data"]["det"]["labels"].detach().cpu().numpy(),
        )

        run.log(
            {
                "map": {
                    "train": {
                        "losses": t_metrics["losses"],
                        "ece": {
                            "prop": train_prop_ece,
                            "det": train_det_ece,
                        },
                    },
                    "val": {
                        "losses": v_metrics["losses"],
                        "ece": {
                            "prop": val_prop_ece,
                            "det": val_det_ece,
                        },
                    },
                },
                "swa": {
                    "train": {
                        "losses": swa_t_metrics["losses"],
                        "ece": {
                            "prop": swa_train_prop_ece,
                            "det": swa_train_det_ece,
                        },
                    },
                    "val": {
                        "losses": swa_v_metrics["losses"],
                        "ece": {
                            "prop": swa_val_prop_ece,
                            "det": swa_val_det_ece,
                        },
                    },
                },
            }
        )

        ## save model
        torch.save(model.state_dict(), os.path.join(model_dir, f"{epoch}_model.pth"))
        torch.save(
            swa_model.state_dict(), os.path.join(model_dir, f"{epoch}_swa_model.pth")
        )
        # break
