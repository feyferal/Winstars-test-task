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

### Train
```bash
python -m Task1.mnist_model.train \
  --epochs 5 \
  --batch_size 64 \
  --lr 0.001 \
  --device cpu
```

### Inference
```bash
python -m Task1.mnist_model.infer \
  --image path/to/image.png \
  --model models/mnist.pth
```

---

## Task 2 — NER + CV Pipeline

### Train NER
```bash
python -m Task2.src.ner.train \
  --model_name bert-base-cased \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --device cpu
```

### Train CV
```bash
python -m Task2.src.img.train \
  --model resnet18 \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.0003 \
  --device cpu
```

---

## Inference (Pipeline)

```bash
python -m Task2.src.pipeline \
  --text "A dog is running in the park" \
  --image path/to/image.jpg \
  --ner_model models/ner \
  --cv_model models/cv.pth \
  --device cpu
```
