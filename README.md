# genpark-auto-annotator

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-zero-success)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

GenPark Auto Annotator is a zero-dependency machine learning data labeling and QA engine implementing spatial bounding box geometric metrics (IoU, GIoU, DIoU), multi-annotator consensus validation via Fleiss' Kappa, and pool-based active learning uncertainty acquisition algorithms.

```mermaid
flowchart LR
    A[Prediction Bounding Boxes] --> B[IoU / GIoU / DIoU Solver]
    B --> C[Spatial Overlap Consensus]
    D[Multi-Rater Confusion Matrix] --> E[Fleiss Kappa Engine]
    E --> F[Annotator Reliability Rating]
    G[Unlabeled Probability Pool] --> H[Shannon Entropy & Margin Sampler]
    H --> I[High-Information Active Learning Queue]
```

## Features
- **Spatial IoU / GIoU / DIoU Geometry**: Calculates overlap area, smallest enclosing convex bounding box, and normalized center distance penalty.
- **Fleiss' Kappa Consensus**: Evaluates categorical agreement across arbitrary numbers of raters and categories with Landis-Koch thresholds.
- **Active Learning Acquisition**: Identifies maximum uncertainty samples using Shannon Entropy, Margin Sampling, and Least Confidence to minimize annotation costs.

## Quickstart

```python
from client import GenParkAutoAnnotator

annotator = GenParkAutoAnnotator()
res = annotator.calculate_bbox_iou_and_giou(
    {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
    {"x1": 20, "y1": 20, "x2": 60, "y2": 60}
)
print("IoU:", res["iou"], "GIoU:", res["giou"])
```

## License
MIT
