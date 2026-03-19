# Multimodal ML Project: MNIST (Task 1) + NER & CV Pipeline (Task 2)

This repository contains two machine learning tasks:

- **Task 1 — MNIST Classification**  
  A baseline computer vision task for digit recognition  

- **Task 2 — Multimodal Pipeline (NER + CV)**  
  A system that combines text processing and image classification  

---

## Task 1: MNIST

A simple image classification task used to:
- validate training pipeline  
- debug model architecture  
- establish a baseline  

**Tech:** PyTorch  

---

## Task 2: NER + CV Pipeline

A multimodal system that processes both **text and images**:

- **NER (Named Entity Recognition)**  
  Extracts animal entities from text  

- **CV (Computer Vision)**  
  Classifies images into animal categories  

- **Pipeline**  
  Combines NER + CV into a unified inference flow  

---

### Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Task 1 — MNIST

### Train+Evaluate
```bash
python -m Task1.mnist.main \
  --algo cnn \
  --epochs 5 \
```

## Task 2 — NER + CV Pipeline

### Train NER
```bash
python -m Task2.src.ner.train \
```

### Train CV
```bash
python -m Task2.src.img.train \
  --epochs 10 \
```

---

## Inference (Pipeline)

```bash
python -m Task2.src.pipeline \
  --text "It is a horse" \
  --image path/to/image.jpg \
```
