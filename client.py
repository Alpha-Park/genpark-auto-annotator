# -*- coding: utf-8 -*-
"""
GenPark Auto Annotator Client
Zero-dependency bounding box IoU / GIoU / DIoU geometry matcher, Fleiss' Kappa multi-annotator
agreement scorer, and active learning uncertainty sampling selector.
"""

import math
from typing import Dict, List, Any, Optional

class GenParkAutoAnnotator:
    """
    Solves bounding box IoU/GIoU geometry, computes Fleiss' Kappa consensus,
    and ranks active learning uncertainty samples.
    """

    def calculate_bbox_iou_and_giou(
        self,
        box1: Dict[str, float],
        box2: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculates Intersection over Union (IoU), Generalized IoU (GIoU), and Distance IoU (DIoU).
        Box format: {x1, y1, x2, y2}
        """
        x1_a, y1_a, x2_a, y2_a = box1["x1"], box1["y1"], box1["x2"], box1["y2"]
        x1_b, y1_b, x2_b, y2_b = box2["x1"], box2["y1"], box2["x2"], box2["y2"]

        # Intersection rect
        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)

        iw = max(0.0, xi2 - xi1)
        ih = max(0.0, yi2 - yi1)
        intersection_area = iw * ih

        # Box areas
        area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
        area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
        union_area = area_a + area_b - intersection_area

        iou = intersection_area / (union_area + 1e-9)

        # Smallest enclosing box C
        xc1 = min(x1_a, x1_b)
        yc1 = min(y1_a, y1_b)
        xc2 = max(x2_a, x2_b)
        yc2 = max(y2_a, y2_b)
        cw = xc2 - xc1
        ch = yc2 - yc1
        enclosing_area = max(1e-9, cw * ch)

        # GIoU = IoU - (Area(C) - Area(Union)) / Area(C)
        giou = iou - ((enclosing_area - union_area) / enclosing_area)

        # DIoU = IoU - (distance between centers)^2 / (diagonal of enclosing box)^2
        cx_a, cy_a = (x1_a + x2_a) / 2.0, (y1_a + y2_a) / 2.0
        cx_b, cy_b = (x1_b + x2_b) / 2.0, (y1_b + y2_b) / 2.0
        center_dist_sq = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2
        diag_c_sq = cw ** 2 + ch ** 2
        diou = iou - (center_dist_sq / max(1e-9, diag_c_sq))

        return {
            "iou": round(iou, 4),
            "giou": round(giou, 4),
            "diou": round(diou, 4),
            "intersection_area": round(intersection_area, 2),
            "union_area": round(union_area, 2),
            "enclosing_box": {"x1": xc1, "y1": yc1, "x2": xc2, "y2": yc2},
            "is_matched_overlap": iou >= 0.50
        }

    def compute_inter_annotator_agreement_kappa(
        self,
        ratings_matrix: List[List[int]],
        num_categories: int
    ) -> Dict[str, Any]:
        """
        Computes Fleiss' Kappa for inter-annotator agreement across multiple categorical raters.
        ratings_matrix: N subjects x k categories count (each row sums to total raters m)
        """
        if not ratings_matrix:
            raise ValueError("Ratings matrix cannot be empty.")

        n_subjects = len(ratings_matrix)
        n_raters = sum(ratings_matrix[0])
        if n_raters <= 1:
            raise ValueError("Total raters per subject must be at least 2.")

        # Compute p_j: proportion of all assignments to j-th category
        total_assignments = n_subjects * n_raters
        p_j = [0.0] * num_categories
        for j in range(num_categories):
            col_sum = sum(ratings_matrix[i][j] for i in range(n_subjects))
            p_j[j] = col_sum / total_assignments

        # Expected agreement P_e_bar = sum(p_j^2)
        p_e_bar = sum(p * p for p in p_j)

        # Compute P_i for each subject: (sum(n_ij^2) - m) / (m * (m - 1))
        p_i = []
        for row in ratings_matrix:
            sq_sum = sum(count * count for count in row)
            pi = (sq_sum - n_raters) / (n_raters * (n_raters - 1))
            p_i.append(pi)

        # Observed agreement P_bar = mean(P_i)
        p_bar = sum(p_i) / n_subjects

        # Fleiss' Kappa: (P_bar - P_e_bar) / (1 - P_e_bar)
        denom = 1.0 - p_e_bar
        kappa = (p_bar - p_e_bar) / denom if abs(denom) > 1e-9 else 1.0
        kappa_val = round(kappa, 4)

        # Interpretation (Landis & Koch)
        if kappa_val < 0.0:
            interpretation = "Poor Agreement"
        elif kappa_val <= 0.20:
            interpretation = "Slight Agreement"
        elif kappa_val <= 0.40:
            interpretation = "Fair Agreement"
        elif kappa_val <= 0.60:
            interpretation = "Moderate Agreement"
        elif kappa_val <= 0.80:
            interpretation = "Substantial Agreement"
        else:
            interpretation = "Almost Perfect Agreement"

        return {
            "fleiss_kappa": kappa_val,
            "interpretation": interpretation,
            "observed_agreement_p_bar": round(p_bar, 4),
            "expected_agreement_p_e_bar": round(p_e_bar, 4),
            "total_subjects": n_subjects,
            "raters_per_subject": n_raters,
            "category_marginals": [round(p, 4) for p in p_j]
        }

    def sample_active_learning_uncertainty(
        self,
        predictions_probabilities: List[Dict[str, Any]],
        sample_size: int = 3,
        strategy: str = "entropy"
    ) -> Dict[str, Any]:
        """
        Calculates active learning acquisition scores (Shannon Entropy, Margin Sampling, Least Confidence).
        """
        scored_candidates = []

        for item in predictions_probabilities:
            sample_id = item.get("id", "sample")
            probs = item.get("probabilities", [])
            if not probs:
                continue

            sorted_p = sorted(probs, reverse=True)
            top1 = sorted_p[0]
            top2 = sorted_p[1] if len(sorted_p) > 1 else 0.0

            # Shannon Entropy: -sum(p * log2(p))
            entropy = -sum(p * math.log2(p + 1e-12) for p in probs if p > 0.0)
            # Margin: top1 - top2 (smaller margin = higher uncertainty)
            margin = top1 - top2
            # Least Confidence: 1 - top1
            least_conf = 1.0 - top1

            score = entropy if strategy == "entropy" else (least_conf if strategy == "least_confidence" else -margin)

            scored_candidates.append({
                "id": sample_id,
                "top_probability": round(top1, 4),
                "entropy": round(entropy, 4),
                "margin": round(margin, 4),
                "least_confidence": round(least_conf, 4),
                "acquisition_priority_score": round(score, 4)
            })

        # Sort descending by priority
        scored_candidates.sort(key=lambda x: x["acquisition_priority_score"], reverse=True)
        selected_samples = scored_candidates[:sample_size]

        return {
            "strategy": strategy,
            "total_pool_size": len(predictions_probabilities),
            "selected_sample_count": len(selected_samples),
            "selected_samples": selected_samples
        }
