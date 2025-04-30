import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as F
import pprint
import math
import numpy as np
import gc
from tqdm import tqdm
import os
import json
from collections import defaultdict

# Transformations with data augmentation
train_transforms = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformations for validation (no augmentation)
val_transforms = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# VOC dataset
def load_voc_datasets(root='./data', train_size=0.8):
    # Full dataset
    full_dataset = datasets.VOCDetection(
        root=root,
        year='2012',
        image_set='train',
        download=False,
        transform=None  # Will apply transforms later
    )
    
    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_size * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create new datasets with appropriate transforms
    train_dataset_transformed = VOCDatasetWithTransform(train_dataset, transform=train_transforms)
    val_dataset_transformed = VOCDatasetWithTransform(val_dataset, transform=val_transforms)
    
    return train_dataset_transformed, val_dataset_transformed

# Custom dataset wrapper to apply transforms
class VOCDatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, annotation = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, annotation


# Parse annotation
def parse_annotations(annotations):
    boxes = []
    labels = []
    for obj in annotations['annotation']['object']:
        bndbox = obj['bndbox']
        xmin = int(bndbox['xmin'])
        ymin = int(bndbox['ymin'])
        xmax = int(bndbox['xmax'])
        ymax = int(bndbox['ymax'])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_index(obj['name']))
    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    return boxes, labels


# VOC class mapping
def class_to_index(class_name):
    class_dict = {
        "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
        "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9,
        "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14,
        "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19
    }
    return class_dict.get(class_name, -1)


# Index to class name mapping
def index_to_class(index):
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    if 0 <= index < len(classes):
        return classes[index]
    return "unknown"


# Custom collate function
def collate_fn(batch):
    images = []
    metadata = []
    targets = []
    for b in batch:
        images.append(b[0])
        boxes, labels = parse_annotations(b[1])
        # Skip samples with no annotations
        if len(boxes) == 0:
            continue
        target = {
            "boxes": boxes,
            "labels": labels,
        }
        targets.append(target)
        metadata.append(b[1])
    
    # Skip batch if all samples had no annotations
    if len(images) == 0:
        return None
        
    images = torch.stack(images, dim=0)
    return images, targets, metadata


# ResNet Backbone
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=True)
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2
        self.stage1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # /4
        self.stage2 = resnet.layer2  # /8
        self.stage3 = resnet.layer3  # /16
        self.stage4 = resnet.layer4  # /32

    def forward(self, x):
        features = []
        x = self.stage0(x); features.append(x)
        x = self.stage1(x); features.append(x)
        x = self.stage2(x); features.append(x)
        x = self.stage3(x); features.append(x)
        x = self.stage4(x); features.append(x)
        return features


# Top-Down FPN-style module
class TopDownModule(nn.Module):
    def __init__(self):
        super(TopDownModule, self).__init__()
        self.conv1x1_M5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.conv1x1_M4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv1x1_M3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv1x1_M2 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv1x1_M1 = nn.Conv2d(64, 256, kernel_size=1)
        self.smooth_conv3x3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, features):
        M5 = self.conv1x1_M5(features[-1])
        upsampled_M5 = F.interpolate(M5, size=(features[-2].shape[2], features[-2].shape[3]), mode='nearest')

        lateral_C4 = self.conv1x1_M4(features[-2])
        M4 = lateral_C4 + upsampled_M5
        M4 = self.smooth_conv3x3(M4)
        upsampled_M4 = F.interpolate(M4, size=(features[-3].shape[2], features[-3].shape[3]), mode='nearest')

        lateral_C3 = self.conv1x1_M3(features[-3])
        M3 = lateral_C3 + upsampled_M4
        M3 = self.smooth_conv3x3(M3)
        upsampled_M3 = F.interpolate(M3, size=(features[-4].shape[2], features[-4].shape[3]), mode='nearest')

        lateral_C2 = self.conv1x1_M2(features[-4])
        M2 = lateral_C2 + upsampled_M3
        M2 = self.smooth_conv3x3(M2)
        upsampled_M2 = F.interpolate(M2, size=(features[0].shape[2], features[0].shape[3]), mode='nearest')

        lateral_C1 = self.conv1x1_M1(features[0])
        M1 = lateral_C1 + upsampled_M2
        M1 = self.smooth_conv3x3(M1)

        return [M1, M2, M3, M4, M5]


# Anchor Generator for FPN
class AnchorGenerator(nn.Module):
    def __init__(self, sizes, aspect_ratios):
        super(AnchorGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
    def generate_anchors(self, feature_map_size, stride, size_idx, device):
        """
        Generate anchors for a single feature map
        
        Args:
            feature_map_size: (height, width) tuple
            stride: Stride for this feature map level
            size_idx: Index to select appropriate anchor size
            device: Device to create tensors on
            
        Returns:
            all_anchors: Tensor of shape (H*W*A, 4) with format [x1, y1, x2, y2]
                where A is the number of anchors per location
        """
        height, width = feature_map_size
        
        # Generate grid of centers
        shifts_x = torch.arange(0, width * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, height * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        # Get the appropriate size for this level
        size = self.sizes[size_idx]
        
        # Generate anchor boxes for each combination of size and aspect ratio
        base_anchors = []
        for aspect_ratio in self.aspect_ratios:
            # Calculate width and height based on size and aspect ratio
            w = size * math.sqrt(aspect_ratio)
            h = size / math.sqrt(aspect_ratio)
            
            # Create base anchor centered at (0, 0)
            x0 = -w / 2
            y0 = -h / 2
            x1 = w / 2
            y1 = h / 2
            
            base_anchors.append([x0, y0, x1, y1])
                
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
        
        # Get number of locations and anchors per location
        num_locations = len(shift_x)
        num_anchors = len(base_anchors)
        
        # Broadcast anchors and shifts to get all anchors
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1).view(-1, 1, 4)
        base_anchors = base_anchors.view(1, num_anchors, 4)
        
        # Apply shifts to base anchors
        all_anchors = shifts + base_anchors
        all_anchors = all_anchors.reshape(-1, 4)
        
        return all_anchors

    def forward(self, feature_maps, image_size):
        """
        Generate anchors for all feature maps
        
        Args:
            feature_maps: List of feature maps (M1, M2, ..., M5)
            image_size: Original image size (height, width)
            
        Returns:
            all_anchors: List of anchor tensors, one per feature map level
        """
        device = feature_maps[0].device
        
        # Get feature map sizes and strides for each level
        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_maps]
        strides = [image_size[0] // size[0] for size in feature_map_sizes]
        
        # For each feature map level, generate anchors
        all_anchors = []
        for level, (size, stride) in enumerate(zip(feature_map_sizes, strides)):
            anchors = self.generate_anchors(size, stride, level, device)
            all_anchors.append(anchors)
            
        return all_anchors


# Detection Heads for classification and bounding box regression
class DetectionHeads(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(DetectionHeads, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        )
        
        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        """
        Forward pass for both classification and regression heads
        
        Args:
            x: Feature map, shape (B, C, H, W)
            
        Returns:
            cls_logits: Classification logits, shape (B, H*W*A, num_classes)
            bbox_preds: Box regression predictions, shape (B, H*W*A, 4)
        """
        batch_size = x.shape[0]
        
        # Apply classification head
        cls_logits = self.cls_head(x)
        # Reshape from (B, A*num_classes, H, W) to (B, H*W*A, num_classes)
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)
        
        # Apply bbox regression head
        bbox_preds = self.bbox_head(x)
        # Reshape from (B, A*4, H, W) to (B, H*W*A, 4)
        bbox_preds = bbox_preds.permute(0, 2, 3, 1).contiguous()
        bbox_preds = bbox_preds.view(batch_size, -1, 4)
        
        return cls_logits, bbox_preds


# Complete FPN Object Detection Model
class FPNObjectDetection(nn.Module):
    def __init__(self, num_classes=20, pretrained=True):
        super(FPNObjectDetection, self).__init__()
        self.backbone = ResNetBackbone()
        self.top_down = TopDownModule()
        
        # Set up anchor generator
        # Use different sizes for different feature map levels
        self.anchor_sizes = [32, 64, 128, 256, 512]  
        self.aspect_ratios = [0.5, 1.0, 2.0]  # Same aspect ratios for all levels
        
        # Calculate number of anchors per location
        num_anchors_per_location = len(self.aspect_ratios)
        
        # Create anchor generator
        self.anchor_generator = AnchorGenerator(
            sizes=self.anchor_sizes,
            aspect_ratios=self.aspect_ratios
        )
        
        # Create detection heads for each pyramid level (M1 to M5)
        self.detection_heads = nn.ModuleList([
            DetectionHeads(256, num_anchors_per_location, num_classes)
            for _ in range(5)  # 5 pyramid levels
        ])
        
        self.num_classes = num_classes
    
    def forward(self, images):
        """
        Forward pass for the complete FPN object detection model
        
        Args:
            images: Input images, shape (B, C, H, W)
            
        Returns:
            cls_logits_list: List of classification logits for each level
            bbox_preds_list: List of bbox prediction deltas for each level
            anchors_list: List of anchors for each level
        """
        # Get image size for anchor generation
        image_size = (images.shape[2], images.shape[3])
        
        # Extract features from backbone
        features = self.backbone(images)
        
        # Apply FPN top-down pathway
        pyramid_features = self.top_down(features)
        
        # Generate anchors for each pyramid level
        anchors_list = self.anchor_generator(pyramid_features, image_size)
        
        # Apply detection heads to each pyramid level
        cls_logits_list = []
        bbox_preds_list = []
        
        for level, (feature, head) in enumerate(zip(pyramid_features, self.detection_heads)):
            cls_logits, bbox_preds = head(feature)
            cls_logits_list.append(cls_logits)
            bbox_preds_list.append(bbox_preds)
        
        return cls_logits_list, bbox_preds_list, anchors_list


# Complete Loss Function Implementation
class DetectionLoss(nn.Module):
    def __init__(self, cls_weight=1.0, bbox_weight=1.0, pos_threshold=0.5, neg_threshold=0.4, 
                 pos_neg_ratio=3, use_focal_loss=True):
        super(DetectionLoss, self).__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.pos_neg_ratio = pos_neg_ratio
        self.use_focal_loss = use_focal_loss
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')
        
    def forward(self, cls_logits_list, bbox_preds_list, anchors_list, targets):
        """
        Calculate loss for detection
        
        Args:
            cls_logits_list: List of classification logits from each level
            bbox_preds_list: List of bbox prediction deltas from each level
            anchors_list: List of anchors from each level
            targets: List of target dictionaries with 'boxes' and 'labels'
            
        Returns:
            loss: Total loss
            cls_loss: Classification loss
            reg_loss: Bounding box regression loss
        """
        batch_size = cls_logits_list[0].shape[0]
        device = cls_logits_list[0].device
        
        # Collect all predicted logits, bbox_preds, and anchors
        all_cls_logits = []
        all_bbox_preds = []
        all_anchors = []
        
        # Number of anchors per level
        num_anchors_per_level = [anchors.shape[0] for anchors in anchors_list]
        
        # Concatenate predictions from all levels
        for cls_logits, bbox_preds, anchors in zip(cls_logits_list, bbox_preds_list, anchors_list):
            all_cls_logits.append(cls_logits)
            all_bbox_preds.append(bbox_preds)
            all_anchors.append(anchors.repeat(batch_size, 1, 1))
        
        # Concat along the anchor dimension (dim=1)
        all_cls_logits = torch.cat(all_cls_logits, dim=1)
        all_bbox_preds = torch.cat(all_bbox_preds, dim=1)
        all_anchors = torch.cat(all_anchors, dim=1)
        
        # Initialize batch losses
        batch_cls_loss = torch.tensor(0.0, device=device)
        batch_reg_loss = torch.tensor(0.0, device=device)
        
        # Process each batch item
        for b in range(batch_size):
            # Get current batch predictions
            cls_logits = all_cls_logits[b]  # Shape: (num_anchors, num_classes)
            bbox_preds = all_bbox_preds[b]  # Shape: (num_anchors, 4)
            anchors = all_anchors[b]  # Shape: (num_anchors, 4)
            
            # Get targets for current batch
            target_boxes = targets[b]["boxes"].to(device)  # Shape: (num_gt, 4)
            target_labels = targets[b]["labels"].to(device)  # Shape: (num_gt)
            
            if len(target_boxes) == 0:
                continue
            
            # Calculate IoU between all anchors and all target boxes
            # Shape: (num_anchors, num_gt)
            ious = box_iou(anchors, target_boxes)
            
            # For each anchor, get the max IoU with any target box and the corresponding target index
            max_ious, max_iou_idxs = ious.max(dim=1)
            
            # Assign positive/negative samples
            # Positives: IoU >= pos_threshold
            pos_mask = max_ious >= self.pos_threshold
            num_pos = pos_mask.sum().item()
            
            # Get assigned target boxes and labels for positive anchors
            # pos_idxs = torch.nonzero(pos_mask, as_tuple=True)[0]
            assigned_targets = target_boxes[max_iou_idxs[pos_mask]]
            assigned_labels = target_labels[max_iou_idxs[pos_mask]]
            
            # Create target tensor for classification
            cls_targets = torch.zeros_like(cls_logits)
            if num_pos > 0:
                cls_targets[pos_mask, assigned_labels] = 1.0
            
            # Negatives: IoU < neg_threshold
            neg_mask = max_ious < self.neg_threshold
            
            # Balance positive and negative samples
            num_neg = min(neg_mask.sum().item(), self.pos_neg_ratio * max(num_pos, 1))
            if num_neg > 0:
                # Get hard negative samples (highest loss among negatives)
                neg_cls_logits = cls_logits[neg_mask]
                neg_probs = torch.sigmoid(neg_cls_logits)
                neg_losses = -(torch.log(1 - neg_probs + 1e-6))
                neg_losses = neg_losses.sum(dim=1)
                _, hard_neg_idxs = neg_losses.topk(num_neg)
                
                # Create a new mask for selected hard negatives
                hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
                hard_neg_mask[neg_mask.nonzero(as_tuple=True)[0][hard_neg_idxs]] = True
                
                # Combined mask for loss calculation
                loss_mask = pos_mask | hard_neg_mask
            else:
                loss_mask = pos_mask
            
            # Calculate classification loss
            if self.use_focal_loss:
                # Focal loss implementation
                alpha = 0.25
                gamma = 2.0
                
                probs = torch.sigmoid(cls_logits)
                pt = torch.where(cls_targets == 1.0, probs, 1 - probs)
                alpha_factor = torch.where(cls_targets == 1.0, alpha, 1 - alpha)
                focal_weight = alpha_factor * (1 - pt).pow(gamma)
                
                bce_loss = F.binary_cross_entropy_with_logits(
                    cls_logits, cls_targets, reduction='none'
                )
                cls_loss = (focal_weight * bce_loss).sum() / max(1, num_pos)
            else:
                # Standard BCE loss
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_logits[loss_mask], cls_targets[loss_mask], reduction='sum'
                ) / max(1, num_pos)
            
            # Calculate regression loss for positive samples only
            if num_pos > 0:
                # Convert anchors and target boxes to deltas
                anchor_widths = anchors[:, 2] - anchors[:, 0]
                anchor_heights = anchors[:, 3] - anchors[:, 1]
                anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
                anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
                
                # Get positive anchors
                pos_anchors = anchors[pos_mask]
                pos_anchor_widths = anchor_widths[pos_mask]
                pos_anchor_heights = anchor_heights[pos_mask]
                pos_anchor_ctr_x = anchor_ctr_x[pos_mask]
                pos_anchor_ctr_y = anchor_ctr_y[pos_mask]
                
                # Get target widths, heights, centers
                target_widths = assigned_targets[:, 2] - assigned_targets[:, 0]
                target_heights = assigned_targets[:, 3] - assigned_targets[:, 1]
                target_ctr_x = assigned_targets[:, 0] + 0.5 * target_widths
                target_ctr_y = assigned_targets[:, 1] + 0.5 * target_heights
                
                # Compute target deltas
                target_dx = (target_ctr_x - pos_anchor_ctr_x) / pos_anchor_widths
                target_dy = (target_ctr_y - pos_anchor_ctr_y) / pos_anchor_heights
                target_dw = torch.log(target_widths / pos_anchor_widths)
                target_dh = torch.log(target_heights / pos_anchor_heights)
                
                target_deltas = torch.stack([target_dx, target_dy, target_dw, target_dh], dim=1)
                
                # Apply normalization
                mean = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                std = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device)
                target_deltas = (target_deltas - mean) / std
                
                # Get predicted deltas for positive anchors
                pred_deltas = bbox_preds[pos_mask]
                
                # Calculate regression loss
                reg_loss = self.smooth_l1(pred_deltas, target_deltas) / max(1, num_pos)
            else:
                reg_loss = torch.tensor(0.0, device=device)
            
            # Add to batch loss
            batch_cls_loss += cls_loss
            batch_reg_loss += reg_loss
        
        # Average over batch size
        batch_cls_loss /= batch_size
        batch_reg_loss /= batch_size
        
        # Weighted sum of losses
        total_loss = self.cls_weight * batch_cls_loss + self.bbox_weight * batch_reg_loss
        
        return total_loss, batch_cls_loss, batch_reg_loss


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: Tensor of shape (N, 4) with format [x1, y1, x2, y2]
        boxes2: Tensor of shape (M, 4) with format [x1, y1, x2, y2]
        
    Returns:
        iou: Tensor of shape (N, M) with IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Compute IoU
    union = area1[:, None] + area2 - intersection
    iou = intersection / (union + 1e-6)
    
    return iou


def decode_boxes(box_deltas, anchors):
    """
    Convert predicted box deltas to absolute box coordinates
    
    Args:
        box_deltas: Tensor of shape (N, 4) with format [dx, dy, dw, dh]
        anchors: Tensor of shape (N, 4) with format [x1, y1, x2, y2]
        
    Returns:
        boxes: Tensor of shape (N, 4) with format [x1, y1, x2, y2]
    """
    # Get anchor centers, widths and heights
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    
    # Apply normalization
    mean = torch.tensor([0.0, 0.0, 0.0, 0.0], device=box_deltas.device)
    std = torch.tensor([0.1, 0.1, 0.2, 0.2], device=box_deltas.device)
    box_deltas = box_deltas * std + mean
    
    # Apply deltas
    dx, dy, dw, dh = box_deltas.unbind(1)
    
    # Convert deltas to absolute coordinates
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    
    # Convert to [x1, y1, x2, y2] format
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h
    
    # Stack predictions
    pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    
    return pred_boxes


# Non-Maximum Suppression (NMS)
def nms(boxes, scores, iou_threshold=0.5):
    """
    Apply non-maximum suppression to avoid detecting multiple boxes for the same object
    
    Args:
        boxes: Tensor of shape (N, 4) with format [x1, y1, x2, y2]
        scores: Tensor of shape (N) with confidence scores
        iou_threshold: IoU threshold for NMS
        
    Returns:
        keep: Tensor of indices of boxes to keep
    """
    # Sort boxes by score
    _, order = torch.sort(scores, descending=True)
    
    keep = []
    while order.numel() > 0:
        # Pick the box with highest score
        i = order[0].item()
        keep.append(i)
        
        # If there's only one box left, break
        if order.numel() == 1:
            break
            
        # Compute IoU of the picked box with the rest
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])
        
        # Remove boxes with IoU over threshold
        mask = ious[0] <= iou_threshold
        order = order[1:][mask]
        
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# Post-processing for inference
def post_process_detections(cls_logits_list, bbox_preds_list, anchors_list, 
                           score_threshold=0.05, nms_threshold=0.5, max_detections=100):
    """
    Post-process detection predictions
    
    Args:
        cls_logits_list: List of classification logits from each level
        bbox_preds_list: List of bbox prediction deltas from each level
        anchors_list: List of anchors from each level
        score_threshold: Minimum score threshold for keeping detections
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections to keep
        
    Returns:
        all_boxes: List (per image) of tensors of shape (N, 4) with format [x1, y1, x2, y2]
        all_scores: List (per image) of tensors of shape (N) with detection scores
        all_labels: List (per image) of tensors of shape (N) with predicted class labels
    """
    batch_size = cls_logits_list[0].shape[0]
    num_classes = cls_logits_list[0].shape[2]
    device = cls_logits_list[0].device
    
    # Prepare lists for batch results
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # Process each image in the batch
    for b in range(batch_size):
        # Concatenate predictions from all feature levels
        boxes_batch = []
        scores_batch = []
        labels_batch = []
        
        # Process each feature level
        for cls_logits, bbox_preds, anchors in zip(cls_logits_list, bbox_preds_list, anchors_list):
            # Get predictions for current batch item
            cls_logits_b = cls_logits[b]  # Shape: (num_anchors, num_classes)
            bbox_preds_b = bbox_preds[b]  # Shape: (num_anchors, 4)
            
            # Apply sigmoid to get probabilities
            scores = torch.sigmoid(cls_logits_b)
            
            # Decode predicted boxes
            boxes = decode_boxes(bbox_preds_b, anchors)
            
            # For each class (except background)
            for cls_idx in range(num_classes):
                # Get scores for this class
                cls_scores = scores[:, cls_idx]
                
                # Filter out low-confidence predictions
                mask = cls_scores > score_threshold
                if mask.sum() == 0:
                    continue
                    
                # Get filtered boxes and scores
                filtered_boxes = boxes[mask]
                filtered_scores = cls_scores[mask]
                
                # Apply NMS
                keep_idxs = nms(filtered_boxes, filtered_scores, nms_threshold)
                
                # Add to batch results
                boxes_batch.append(filtered_boxes[keep_idxs])
                scores_batch.append(filtered_scores[keep_idxs])
                labels_batch.append(torch.full_like(keep_idxs, cls_idx))
        
        # Concatenate results from all classes
        if len(boxes_batch) > 0:
            boxes_batch = torch.cat(boxes_batch, dim=0)
            scores_batch = torch.cat(scores_batch, dim=0)
            labels_batch = torch.cat(labels_batch, dim=0)
            
            # Keep top-k detections
            if len(scores_batch) > max_detections:
                _, top_k_idxs = torch.topk(scores_batch, max_detections)
                boxes_batch = boxes_batch[top_k_idxs]
                scores_batch = scores_batch[top_k_idxs]
                labels_batch = labels_batch[top_k_idxs]
        else:
            # No detections found
            boxes_batch = torch.zeros((0, 4), device=device)
            scores_batch = torch.zeros(0, device=device)
            labels_batch = torch.zeros(0, dtype=torch.long, device=device)
            
        all_boxes.append(boxes_batch)
        all_scores.append(scores_batch)
        all_labels.append(labels_batch)
    
    return all_boxes, all_scores, all_labels


# Training function
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train the model for one epoch
    
    Args:
        model: Detection model
        optimizer: Optimizer
        data_loader: Data loader
        device: Device to use
        epoch: Current epoch
        print_freq: Print frequency
    """
    model.train()
    
    # Create loss function
    criterion = DetectionLoss(cls_weight=1.0, bbox_weight=1.0, use_focal_loss=True)
    
    # Tracking metrics
    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    
    # Progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for i, batch in enumerate(pbar):
        # Skip batch if it's None
        if batch is None:
            continue
            
        # Unpack batch
        images, targets, _ = batch
        images = images.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        cls_logits_list, bbox_preds_list, anchors_list = model(images)
        
        # Compute loss
        loss, cls_loss, reg_loss = criterion(cls_logits_list, bbox_preds_list, anchors_list, targets)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Update running losses
        running_loss += loss.item()
        running_cls_loss += cls_loss.item()
        running_reg_loss += reg_loss.item()
        
        # Update progress bar description
        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / (i + 1)
            avg_cls_loss = running_cls_loss / (i + 1)
            avg_reg_loss = running_reg_loss / (i + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'cls_loss': f'{avg_cls_loss:.4f}',
                'reg_loss': f'{avg_reg_loss:.4f}'
            })
    
    # Final metrics for the epoch
    avg_loss = running_loss / len(pbar)
    avg_cls_loss = running_cls_loss / len(pbar)
    avg_reg_loss = running_reg_loss / len(pbar)
    
    return avg_loss, avg_cls_loss, avg_reg_loss


# Validation function
def validate(model, data_loader, device):
    """
    Validate the model
    
    Args:
        model: Detection model
        data_loader: Data loader
        device: Device to use
        
    Returns:
        val_loss: Validation loss
    """
    model.eval()
    
    # Create loss function
    criterion = DetectionLoss(cls_weight=1.0, bbox_weight=1.0, use_focal_loss=True)
    
    # Tracking metrics
    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    
    # Progress bar
    pbar = tqdm(data_loader, desc="Validation")
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            # Skip batch if it's None
            if batch is None:
                continue
                
            # Unpack batch
            images, targets, _ = batch
            images = images.to(device)
            
            # Forward
            cls_logits_list, bbox_preds_list, anchors_list = model(images)
            
            # Compute loss
            loss, cls_loss, reg_loss = criterion(cls_logits_list, bbox_preds_list, anchors_list, targets)
            
            # Update running losses
            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            running_reg_loss += reg_loss.item()
    
    # Calculate average losses
    avg_loss = running_loss / len(pbar)
    avg_cls_loss = running_cls_loss / len(pbar)
    avg_reg_loss = running_reg_loss / len(pbar)
    
    print(f"Validation: loss={avg_loss:.4f}, cls_loss={avg_cls_loss:.4f}, reg_loss={avg_reg_loss:.4f}")
    
    return avg_loss


# Inference function for visualizing predictions
def visualize_predictions(model, images, score_threshold=0.5, nms_threshold=0.5):
    """
    Run inference and visualize predictions
    
    Args:
        model: Detection model
        images: Images tensor of shape (B, C, H, W)
        score_threshold: Score threshold for filtering detections
        nms_threshold: IoU threshold for NMS
        
    Returns:
        fig: Figure with detection visualizations
    """
    device = next(model.parameters()).device
    images = images.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    with torch.no_grad():
        cls_logits_list, bbox_preds_list, anchors_list = model(images)
        boxes, scores, labels = post_process_detections(
            cls_logits_list, bbox_preds_list, anchors_list,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold
        )
    
    # Create figure for visualization
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
    if len(images) == 1:
        axes = [axes]
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Plot each image
    for i, (img, img_boxes, img_scores, img_labels) in enumerate(zip(images, boxes, scores, labels)):
        # Denormalize and convert to numpy
        img = img.cpu().permute(1, 2, 0)
        img = img * std + mean
        img = img.numpy().clip(0, 1)
        
        # Display image
        axes[i].imshow(img)
        
        # Plot boxes
        for box, score, label in zip(img_boxes.cpu(), img_scores.cpu(), img_labels.cpu()):
            x1, y1, x2, y2 = box.tolist()
            class_name = index_to_class(label.item())
            color = plt.cm.hsv(label.item() / 20)
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor=color, linewidth=2)
            axes[i].add_patch(rect)
            
            # Add label
            axes[i].text(x1, y1, f'{class_name}: {score:.2f}', 
                        bbox=dict(facecolor=color, alpha=0.5), 
                        fontsize=8, color='white')
        
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# Main training loop
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset, val_dataset = load_voc_datasets()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    model = FPNObjectDetection(num_classes=20)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    # Number of epochs
    num_epochs = 1
    
    # Training history
    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_reg_loss': [],
        'val_loss': []
    }
    
    # Path to save model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Best validation loss
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch+1
        )
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_reg_loss'].append(train_reg_loss)
        history['val_loss'].append(val_loss)
        
        # Save model if it's the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(model_dir, 'best_model.pth'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'history': history,
        }, os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Overall Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_cls_loss'], label='Classification Loss')
        plt.plot(history['train_reg_loss'], label='Regression Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Component Losses')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_history.png'))
        plt.close()
    
    print("Training complete!")
    
    # Load best model for testing
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['loss']:.4f}")
    
    # Run inference on a few validation samples
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is not None:
                images, targets, metadata = batch
                fig = visualize_predictions(model, images)
                fig.savefig(os.path.join(model_dir, f'sample_prediction_{i}.png'))
                plt.close(fig)
                if i >= 4:  # Just visualize a few samples
                    break


if __name__ == "__main__":
    main()