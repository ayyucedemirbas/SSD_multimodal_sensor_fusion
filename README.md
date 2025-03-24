# SSD_multimodal_sensor_fusion


## lidar_to_bev

**Purpose:**
This function converts a LiDAR point cloud into a Bird’s Eye View (BEV) representation. The BEV grid is of size 512×512 pixels covering a specified physical range (in meters).

**Channels:**
It generates three channels:
- Density: Number of LiDAR points per grid cell.
- Height: Maximum height value per cell.
- Intensity: Sum of intensities per cell, normalized appropriately.

## radar_to_map
**Purpose:**
Similar to LiDAR processing, this function projects radar point clouds onto a grid but returns two channels:
- Range: Maximum range values per grid cell.
- Velocity: Averaged velocities per grid cell after proper normalization.

## AiMotiveSSD_Dataset

**Dataset Initialization:**
The dataset class wraps the aiMotive dataset and allows switching between training and validation splits.

**Data Retrieval:**
For each sample:
- LiDAR data is processed to BEV.
- Radar data from the front and back is processed to a map.
- The resulting sensor data are concatenated along the channel dimension.
- Annotations are converted using the defined category mapping.

**Return Value:**
Each item returns a fused sensor tensor (multi-channel image-like tensor) and corresponding object annotations.

### prepare_lidar_data & prepare_radar_data:
These helper methods call the earlier defined conversion functions to get BEV maps for LiDAR and radar respectively.

### get_targets:
This method loops over object annotations and extracts:
- Location ```(x, y)``` and dimensions ```(l, w)```.
- Orientation (using quaternion component).
- Velocities ```(vel_x, vel_y)```.
- Category index (using the mapping).

The extracted values are converted into a tensor per object. Finally, all object tensors are stacked into one tensor; if there are no objects, a dummy tensor is returned.

## SSDDetector
This simple SSD detector is defined with:

**Input Channels:**
The fused sensor data have 7 channels (3 from LiDAR BEV and 2 each from two radar maps).

**Base Network:**
A small CNN with two convolutional layers and pooling to extract features.

**Detection Heads:**
Two convolutional layers are used:

**Localization head (loc_head):** Predicts offsets for bounding boxes (4 values per box).

**Classification head (cls_head):** Predicts class scores for each default box.

**Output Reshaping:**
The output tensors are permuted and reshaped so that they have shape ```(B, num_boxes, 4)``` and ```(B, num_boxes, num_classes)``` respectively.

## generate_default_boxes
**Default Boxes:** This function generates anchor boxes for each cell in a feature map. These anchors serve as priors for the detector.

**Parameters:** Default scales and aspect ratios are predefined. Each cell in the feature map is associated with 4 boxes.

## convert_gt_boxes
Converts ground truth bounding boxes from physical coordinates into pixel coordinates on the BEV map.

Returns boxes in the ```(center x, center y, width, height)``` format and the corresponding labels.

## compute_iou
This function calculates the overlap between predicted boxes and ground truth boxes

## match_anchors
**Purpose:**
Matches each default box with the ground truth box that has the highest IoU. Boxes with IoU above a threshold (0.5) are considered positive.

**Offset Computation:**
For positive matches, calculates the offset between the default box and ground truth box (which the model learns to predict).

**Targets:**
Returns localization targets (offsets) and classification targets (labels).

## custom_collate_fn
**Dataset Initialization:**
Instantiates the training dataset using the provided root directory.

**Custom Collate Function:**
Since the number of objects varies per image, a custom collate function is used to keep targets as a list.

**DataLoader:**
Creates a PyTorch DataLoader to load the training data in batches.

##  Model Initialization, Loss, and Optimizer Setup
**Model:**
The SSDDetector is instantiated with the expected number of input channels (7) and classes (4).

**Loss Functions:**
Smooth L1 loss for bounding box regression (localization) and cross-entropy loss for classification.

**Optimizer:**
Adam optimizer is used with a learning rate of 1e-4.

**Default Boxes:**
The anchor boxes are generated and moved to the same device.

## Training Loop

**Epoch Loop:**
The script trains for 10 epochs.

**Batch Processing:**
For each batch, it:
- Moves the fused sensor data to the device.

- Processes each sample’s ground truth to convert physical coordinates into pixel coordinates.

- Matches ground truth boxes with default boxes to compute regression (offsets) and classification targets.

- Stacks targets for the entire batch.

**Forward Pass & Loss Computation:**
The model outputs predicted bounding box offsets and class scores, which are compared with the targets using the defined loss functions.

**Backpropagation:**
The loss is backpropagated and the model parameters are updated.

**Logging:**
Loss is printed every few steps.
