# Local model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.494 | 0.359     | 0.523    | 0.925           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.437 | 0.398     | 0.565    | 0.919           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.465 | 0.398     | 0.564    | 0.920           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.445 | 0.396     | 0.561    | 0.921           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.476 | 0.373     | 0.538    | 0.923           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.466 | 0.372     | 0.536    | 0.916           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.440 | 0.408     | 0.574    | 0.929           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.472 | 0.380     | 0.546    | 0.918           |

# Global model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.350 | 0.504     | 0.668    | 0.933           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.325 | 0.513     | 0.676    | 0.934           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.245 | 0.631     | 0.772    | 0.953           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.331 | 0.509     | 0.672    | 0.933           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.391 | 0.458     | 0.627    | 0.921           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.370 | 0.463     | 0.631    | 0.926           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.241 | 0.632     | 0.772    | 0.953           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.334 | 0.505     | 0.669    | 0.932           |

# Global-Local model

### U-Net

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.357 | 0.495     | 0.658    | 0.932           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.345 | 0.498     | 0.662    | 0.929           |

### Linknet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.257 | 0.618     | 0.763    | 0.949           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.264 | 0.588     | 0.738    | 0.944           |

### PSPNet

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.374 | 0.476     | 0.642    | 0.927           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.276 | 0.569     | 0.724    | 0.945           |

### FPN

##### Sigmoid

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.261 | 0.612     | 0.758    | 0.948           |

##### Relu

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.247 | 0.610     | 0.756    | 0.951           |

# Global-Local models with mixed architectures

### Global: unet, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.273 | 0.602     | 0.749    | 0.949           |

### Global: unet, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.267 | 0.607     | 0.753    | 0.947           |

### Global: unet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.260 | 0.613     | 0.757    | 0.950           |

### Global: linknet, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.238 | 0.633     | 0.774    | 0.951           |

### Global: linknet, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.268 | 0.612     | 0.757    | 0.949           |

### Global: linknet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.255 | 0.613     | 0.758    | 0.948           |

### Global: pspnet, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.307 | 0.554     | 0.711    | 0.936           |

### Global: pspnet, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.282 | 0.576     | 0.729    | 0.940           |

### Global: pspnet, Local: fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.295 | 0.580     | 0.732    | 0.942           |

### Global: fpn, Local: unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.270 | 0.605     | 0.752    | 0.949           |

### Global: fpn, Local: linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.301 | 0.585     | 0.735    | 0.944           |

### Global: fpn, Local: pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.260 | 0.627     | 0.769    | 0.952           |

# Global-Global models with mixed architectures

### Global-Global: unet, linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.253 | 0.613     | 0.759    | 0.948           |

### Global-Global: unet, pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.345 | 0.499     | 0.662    | 0.935           |

### Global-Global: unet, fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.326 | 0.521     | 0.683    | 0.934           |

### Global-Global: linknet, unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.269 | 0.597     | 0.746    | 0.945           |

### Global-Global: linknet, pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.247 | 0.624     | 0.767    | 0.951           |

### Global-Global: linknet, fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.244 | 0.624     | 0.766    | 0.952           |

### Global-Global: pspnet, unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.268 | 0.597     | 0.746    | 0.946           |

### Global-Global: pspnet, linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.276 | 0.598     | 0.747    | 0.947           |

### Global-Global: pspnet, fpn

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.262 | 0.618     | 0.763    | 0.949           |

### Global-Global: fpn, unet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.241 | 0.626     | 0.767    | 0.952           |

### Global-Global: fpn, linknet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.254 | 0.614     | 0.760    | 0.948           |

### Global-Global: fpn, pspnet

| loss  | iou_score | f1-score | binary_accuracy |
|-------|-----------|----------|-----------------|
| 0.246 | 0.623     | 0.766    | 0.950           |

