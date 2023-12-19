# Local model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.615 | 0.262     | 0.411    | 0.971           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.680 | 0.193     | 0.321    | 0.949           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.700 | 0.203     | 0.335    | 0.957           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.657 | 0.217     | 0.354    | 0.959           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.646 | 0.231     | 0.372    | 0.964           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.661 | 0.209     | 0.343    | 0.961           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.646 | 0.231     | 0.372    | 0.967           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.718 | 0.183     | 0.307    | 0.949           |

# Global model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.669 | 0.214     | 0.350    | 0.958           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.644 | 0.221     | 0.357    | 0.962           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.590 | 0.276     | 0.428    | 0.969           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.599 | 0.260     | 0.408    | 0.968           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.744 | 0.155     | 0.265    | 0.930           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.675 | 0.197     | 0.326    | 0.950           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.633 | 0.238     | 0.380    | 0.963           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.589 | 0.267     | 0.418    | 0.972           |

# Global-Local model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.679 | 0.201     | 0.330    | 0.949           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.667 | 0.212     | 0.345    | 0.957           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.702 | 0.181     | 0.304    | 0.948           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.637 | 0.228     | 0.367    | 0.960           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.707 | 0.181     | 0.304    | 0.941           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.691 | 0.185     | 0.309    | 0.950           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.663 | 0.215     | 0.349    | 0.958           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.601 | 0.259     | 0.404    | 0.966           |

# Global-Local models with mixed architectures

### Global: unet, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.641 | 0.234     | 0.374    | 0.961           |

### Global: unet, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.668 | 0.211     | 0.345    | 0.953           |

### Global: unet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.611 | 0.256     | 0.402    | 0.967           |

### Global: linknet, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.612 | 0.251     | 0.396    | 0.964           |

### Global: linknet, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.620 | 0.262     | 0.409    | 0.967           |

### Global: linknet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.644 | 0.228     | 0.365    | 0.963           |

### Global: pspnet, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.722 | 0.166     | 0.282    | 0.937           |

### Global: pspnet, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.692 | 0.188     | 0.313    | 0.952           |

### Global: pspnet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.670 | 0.214     | 0.349    | 0.955           |

### Global: fpn, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.670 | 0.209     | 0.341    | 0.957           |

### Global: fpn, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.707 | 0.192     | 0.318    | 0.949           |

### Global: fpn, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.650 | 0.235     | 0.378    | 0.967           |

