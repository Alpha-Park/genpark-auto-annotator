# -*- coding: utf-8 -*-
"""
GenPark Auto Annotator FastMCP Server
"""

import json
from typing import Dict, List, Any, Optional
from mcp.server.fastmcp import FastMCP
from client import GenParkAutoAnnotator

mcp = FastMCP("genpark-auto-annotator")
annotator = GenParkAutoAnnotator()

@mcp.tool()
def calculate_bbox_iou_and_giou(
    box1_json: str,
    box2_json: str
) -> str:
    """
    Calculates Intersection over Union (IoU), Generalized IoU (GIoU), and Distance IoU (DIoU).
    """
    b1 = json.loads(box1_json)
    b2 = json.loads(box2_json)
    res = annotator.calculate_bbox_iou_and_giou(b1, b2)
    return json.dumps(res, indent=2)

@mcp.tool()
def compute_inter_annotator_agreement_kappa(
    ratings_matrix_json: str,
    num_categories: int
) -> str:
    """
    Computes Fleiss' Kappa for inter-annotator agreement across multiple categorical raters.
    """
    matrix = json.loads(ratings_matrix_json)
    res = annotator.compute_inter_annotator_agreement_kappa(matrix, num_categories)
    return json.dumps(res, indent=2)

@mcp.tool()
def sample_active_learning_uncertainty(
    predictions_probabilities_json: str,
    sample_size: int = 3,
    strategy: str = "entropy"
) -> str:
    """
    Calculates active learning acquisition scores (Shannon Entropy, Margin Sampling, Least Confidence).
    """
    pool = json.loads(predictions_probabilities_json)
    res = annotator.sample_active_learning_uncertainty(pool, sample_size, strategy)
    return json.dumps(res, indent=2)

def main():
    print("Testing Auto Annotator:")
    b1 = {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
    b2 = {"x1": 20, "y1": 20, "x2": 60, "y2": 60}
    print(calculate_bbox_iou_and_giou(json.dumps(b1), json.dumps(b2)))

if __name__ == "__main__":
    main()
