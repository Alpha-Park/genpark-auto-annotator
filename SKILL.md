---
name: genpark-auto-annotator
description: GenPark Auto Annotator - Bounding box IoU / GIoU / DIoU geometry matcher, Fleiss' Kappa multi-annotator agreement scorer, and active learning uncertainty sampler with zero dependencies.
category: Service
version: 1.0.0
license: MIT
tools:
  - calculate_bbox_iou_and_giou
  - compute_inter_annotator_agreement_kappa
  - sample_active_learning_uncertainty
---

# Genpark Auto Annotator

## Overview & Agent Behavioral Directives
You are an autonomous agent specialized in **Genpark Auto Annotator**.
Your primary directive is to execute: GenPark Auto Annotator - Bounding box IoU / GIoU / DIoU geometry matcher, Fleiss' Kappa multi-annotator agreement scorer, and active learning uncertainty sampler with zero dependencies.
Always ensure mathematical precision, parameter validation, and deterministic outputs.

## When to Activate This Skill
- Activate when an agent workflow or user instruction requires:
- `calculate_bbox_iou_and_giou`
- `compute_inter_annotator_agreement_kappa`
- `sample_active_learning_uncertainty`
- Activate when operating in zero-dependency environments requiring pure Python standard library execution.

## How to Execute

### Option 1: Direct Python Execution (Zero-Dependency)
```python
from client import *

# Client operates purely on standard Python library:
# Initialized and ready for direct function invocation.
```

### Option 2: FastMCP Tool Calling
Tools registered in `mcp_server.py`:
- `calculate_bbox_iou_and_giou`
- `compute_inter_annotator_agreement_kappa`
- `sample_active_learning_uncertainty`

## Core Mathematical & Algorithmic Guardrails
- **Zero-Dependency Policy**: 100% Python standard library only (`math`, `heapq`, `hashlib`, `urllib`, `collections`, `itertools`, `typing`).
- **Zero-Mock Policy**: Pure deterministic algorithms with full validation and error bounds.
