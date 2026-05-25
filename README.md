# AI Support Ticket Auto Tagger

## Objective of the Task

The objective of this project is to automatically classify free-text customer support tickets into relevant categories using a Large Language Model (LLM).

This application analyzes support ticket subject and description, then predicts the **top 3 most probable tags** with confidence scores and explanations.

The project also demonstrates and compares:

- **Zero-shot learning** (classification without examples)
- **Few-shot learning** (classification using prompt examples)

The goal is to evaluate how prompt engineering improves LLM-based text classification performance.

---

## Methodology / Approach

### 1. LLM-Based Ticket Classification
This project uses the **Groq API** with the **LLaMA 3.3 70B Versatile** model to perform intelligent support ticket tagging.

Each ticket is analyzed and classified into predefined categories such as:

- Technical issue
- Billing inquiry
- Refund request
- Account access
- Software bug
- Cancellation request
- Payment failure
- Product inquiry
- Performance issue
- Hardware failure
- Login/authentication

---

### 2. Prompt Engineering

Two different prompting strategies are implemented:

#### Zero-Shot Prompting
The model receives:

- List of available tags
- Ticket subject
- Ticket description
- Instructions to return top 3 predictions

No prior examples are provided.

---

#### Few-Shot Prompting
The model receives:

- List of available tags
- Multiple labeled ticket examples
- New ticket for classification

These examples help guide the model toward more accurate predictions.

---

### 3. Output Format

For every ticket, the model returns:

- **Top 3 predicted tags**
- **Confidence score (0–100%)**
- **Reasoning for each prediction**

Example:

```json
{
  "tags": [
    {
      "tag": "Login/authentication",
      "confidence": 85,
      "reason": "User unable to access account"
    }
  ]
}
```

---

### 4. Comparison Module

The application compares:

- Zero-shot prediction
- Few-shot prediction
- Processing time
- Accuracy against ground truth (if provided)

This helps evaluate which prompting strategy performs better.

---



## Key Results / Observations

### Prompt Engineering Improves Accuracy

Few-shot prompting generally performs better than zero-shot prompting because example tickets provide contextual guidance to the model.

---

### Top-3 Predictions Improve Reliability

Returning multiple ranked tags improves classification robustness and better reflects uncertainty in ambiguous tickets.

---

### LLMs Handle Unstructured Text Well

The model successfully understands:

- informal customer language
- incomplete descriptions
- mixed technical/business issues

without traditional feature engineering.

---

### Batch Evaluation Enables Performance Benchmarking

Using dataset evaluation helps measure:

- model consistency
- overall accuracy
- prompt effectiveness

which is essential for real-world deployment.

---

## Features

- Flask web application
- Modern responsive UI
- Single ticket classification
- Zero-shot vs Few-shot comparison
- Batch CSV evaluation
- JSON output parsing and validation
- Retry handling for API failures
- Confidence score visualization
- Accuracy measurement dashboard

---

## Tech Stack

- **Python**
- **Flask**
- **Groq API**
- **LLaMA 3.3 70B**
- **HTML/CSS/JavaScript**
- **Bootstrap 5**
- **Pandas**

---

## How to Run

### Install dependencies

```bash
pip install flask groq pandas
```

---

### Set API key

```bash
export GROQ_API_KEY=your_api_key
```

Windows:

```bash
set GROQ_API_KEY=your_api_key
```

---

### Run application

```bash
python app.py
```

---

### Open browser

```bash
http://127.0.0.1:5000
```

---

## Skills Gained

- Prompt engineering
- LLM-based text classification
- Zero-shot learning
- Few-shot learning
- Multi-class prediction and ranking
- API integration
- Flask web development
- Dataset evaluation
- Model performance comparison

---
