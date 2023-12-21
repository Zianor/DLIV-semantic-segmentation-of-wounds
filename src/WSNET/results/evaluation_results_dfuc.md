# Local model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.627 | 0.252     | 0.399    | 0.972           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.685 | 0.189     | 0.315    | 0.951           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.707 | 0.199     | 0.330    | 0.960           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.660 | 0.216     | 0.352    | 0.962           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.652 | 0.227     | 0.366    | 0.966           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.668 | 0.204     | 0.336    | 0.963           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.650 | 0.230     | 0.369    | 0.970           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.729 | 0.176     | 0.297    | 0.951           |

# Global model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.682 | 0.205     | 0.336    | 0.959           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.654 | 0.214     | 0.346    | 0.964           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.607 | 0.263     | 0.411    | 0.970           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.618 | 0.245     | 0.389    | 0.969           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.751 | 0.150     | 0.258    | 0.932           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.688 | 0.187     | 0.312    | 0.951           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.658 | 0.220     | 0.355    | 0.963           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.605 | 0.256     | 0.402    | 0.973           |

# Global-Local model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.692 | 0.192     | 0.317    | 0.951           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.686 | 0.198     | 0.325    | 0.957           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.721 | 0.168     | 0.285    | 0.948           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.662 | 0.210     | 0.342    | 0.960           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.720 | 0.172     | 0.290    | 0.942           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.702 | 0.177     | 0.298    | 0.951           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.684 | 0.200     | 0.328    | 0.959           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.614 | 0.249     | 0.391    | 0.967           |

# Global-Local models with mixed architectures

### Global: unet, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.656 | 0.222     | 0.358    | 0.962           |

### Global: unet, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.685 | 0.199     | 0.327    | 0.954           |

### Global: unet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.633 | 0.239     | 0.379    | 0.967           |

### Global: linknet, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.630 | 0.237     | 0.379    | 0.965           |

### Global: linknet, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.640 | 0.245     | 0.387    | 0.968           |

### Global: linknet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.662 | 0.215     | 0.347    | 0.964           |

### Global: pspnet, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.733 | 0.158     | 0.271    | 0.938           |

### Global: pspnet, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.717 | 0.171     | 0.288    | 0.951           |

### Global: pspnet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.684 | 0.204     | 0.335    | 0.956           |

### Global: fpn, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.686 | 0.198     | 0.325    | 0.958           |

### Global: fpn, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.725 | 0.178     | 0.298    | 0.949           |

### Global: fpn, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.673 | 0.218     | 0.353    | 0.967           |

