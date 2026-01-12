# Sentiment Analysis of Internet Memes: Classical Multimodal Methods vs Vision-Language Large Models

## Project Overview

This repository contains the implementation and experimental framework for a **Master’s thesis** focused on **sentiment analysis of internet memes**.  
The main objective of the project is to **compare classical multimodal sentiment analysis approaches** with **modern Vision-Language Large Models (VLLMs)** and analyze their effectiveness, limitations, and behavior across different types of memes.

Internet memes are a challenging multimodal domain, combining **visual content, embedded text, humor, personal relevance, surprising elements, and cultural context**. While classical approaches rely on explicit feature extraction and model fusion, VLLMs aim to perform end-to-end reasoning over both modalities.

---

## Research Objectives

- Build a **classical multimodal sentiment analysis pipeline** for memes using separate vision and language models
- Compare classical methods with **state-of-the-art Vision-Language Large Models**
- Perform a **fine-grained behavioral analysis of VLLMs**
- Evaluate model robustness to relatability, offensiveness, humor, and unexpectedness content
- Develop a **proof-of-concept web application** for real-time meme sentiment estimation

---

## Project Structure

The project is divided into **two main research parts**, both reflected in this repository.

---

## Part I — Classical Multimodal Sentiment Analysis

This part focuses on building a traditional multimodal representation and fusion pipeline.

### Methodology

- **Text Encoder**
  - BERT-based model for extracting textual representations from meme captions
- **Image Encoder**
  - ResNet-based convolutional neural network for visual feature extraction
- **Multimodal Fusion Strategies**
  - Early fusion
  - Late fusion
  - Feature-level concatenation followed by classification

### Goals

- Establish a strong classical baseline for meme sentiment analysis
- Analyze the strengths and limitations of explicit multimodal fusion
- Provide a reference point for comparison with VLLM-based approaches

---

## Part II — Vision-Language Large Models (VLLMs)

This part is dedicated to an in-depth study of modern Vision-Language Large Models and their behavior in meme sentiment analysis tasks.

### Scope of Analysis

- Evaluation of multiple Vision-Language Large Models 
- Prompt engineering and output interpretation strategies
- Analysis of model sensitivity to different meme characteristics:
  - Humor
  - Relatability 
  - Unexpectedness 
  - Offensive content
- Comparison between zero-shot and few-shot inference

### Output Representation

Instead of a single sentiment label, VLLMs are prompted to estimate **probabilistic sentiment distributions**, such as:

- Offensive (%)
- Funny (%)
- Personal (%)
- Surprising (%)

---

## Proof of Concept — Web Application

As the final stage of the project, a **proof-of-concept web application** is developed.

### Features

- Upload an image containing a meme
- Automatic sentiment analysis using a selected VLLM
- Visualization of sentiment scores in the form of charts
- Interactive comparison between different models (optional)

This application demonstrates the **practical applicability of VLLMs** for real-world multimodal sentiment analysis.

---

## Technologies Used

- Python
- PyTorch
- Transformers (Hugging Face)
- Vision models (ResNet)
- Vision-Language Large Models (VLLMs)
- Frontend visualization (Gradio)

## Authors

**Katarzyna Michalska**  
**Matwej Novgorodtsev**  

Master’s Thesis Project  
Universitet of Adam Mickiewicz in Poznan

---
