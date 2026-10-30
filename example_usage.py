# -*- coding: utf-8 -*-
import json
from client import GenParkAutoAnnotator

annotator = GenParkAutoAnnotator()

# 1. Bounding Box IoU / GIoU / DIoU Calculation
boxA = {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 300.0}
boxB = {"x1": 150.0, "y1": 120.0, "x2": 320.0, "y2": 280.0}
geo = annotator.calculate_bbox_iou_and_giou(boxA, boxB)
print(f"Bounding Box Geometry: IoU={geo['iou']} | GIoU={geo['giou']} | DIoU={geo['diou']} (Matched={geo['is_matched_overlap']})")

# 2. Fleiss' Kappa Inter-Annotator Agreement
# Matrix: 5 subjects, evaluated by 14 raters across 3 categories (e.g. Low, Medium, High)
ratings = [
    [0, 0, 14],
    [0, 2, 12],
    [0, 0, 14],
    [3, 10, 1],
    [1, 4, 9]
]
agreement = annotator.compute_inter_annotator_agreement_kappa(ratings, num_categories=3)
print()
print(f"Annotator Agreement: Fleiss Kappa={agreement['fleiss_kappa']} ({agreement['interpretation']}) | Observed={agreement['observed_agreement_p_bar']}")

# 3. Active Learning Uncertainty Sampling
unlabeled_pool = [
    {"id": "doc_101", "probabilities": [0.92, 0.05, 0.03]}, # confident
    {"id": "doc_102", "probabilities": [0.45, 0.42, 0.13]}, # high entropy / narrow margin
    {"id": "doc_103", "probabilities": [0.34, 0.33, 0.33]}, # maximum entropy
    {"id": "doc_104", "probabilities": [0.75, 0.15, 0.10]}
]
al_selection = annotator.sample_active_learning_uncertainty(unlabeled_pool, sample_size=2, strategy="entropy")
print()
print(f"Active Learning ({al_selection['strategy']}) Sampled Top {al_selection['selected_sample_count']} Items for Human Review:")
for item in al_selection["selected_samples"]:
    print(f"  {item['id']}: Entropy={item['entropy']} | Margin={item['margin']} | Priority={item['acquisition_priority_score']}")
