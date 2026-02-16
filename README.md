
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-4.2.0-blue.svg)](#)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](#)
[![Production](https://img.shields.io/badge/Production-Live-brightgreen.svg?logo=amazonaws)](https://snake.aws.monce.ai)
[![API](https://img.shields.io/badge/API-snake.aws.monce.ai-blue.svg?logo=fastapi)](https://snake.aws.monce.ai/health)
[![Algorithm](https://img.shields.io/badge/Algorithm-SAT--based-purple.svg)](#)
[![Explainability](https://img.shields.io/badge/XAI-Fully_Explainable-orange.svg)](#)
[![Complexity](https://img.shields.io/badge/Complexity-O(n·log(n)·m·b²)-lightgrey.svg)](#)
[![Architecture](https://img.shields.io/badge/Matching-Snake_%2B_Fuzzy_%2B_LLM-red.svg)](#)

**Author:** Charles Dana · [Algorithme.ai](https://algorithme.ai)

# Algorithme AI - Snake 🐍
**Author:** Charles Dana  
**Date:** December 2025  
**Complexity:** O(n·log(n)·m·bucket²)

A Python library for CSV data analysis and classification using a SAT logic-based approach with explanatory clauses.

Snake is an **XAI (Explainable AI)** polynomial-time multiclass oracle. It provides high-accuracy classification while maintaining a full "Audit Trail" for every prediction, allowing you to understand *why* the model reached a specific conclusion through "lookalike" analysis and logical AND statements.

---

### 🚀 Performance
The Snake algorithm achieves a high accuracy score of **0.81100** on the **Titanic Kaggle Dataset** challenge.

### 🔍 Explainability (XAI)
Unlike "black-box" models, Snake provides a full audit of its reasoning. It identifies "lookalikes" from the training set and displays the exact logical conditions that link them to the new data point.

**Example Audit Output:**
> **Predicted outcome:** Class [0] (98.88% probability)  
> **Reasoning:** Datapoint is a lookalike to Passenger #5 (Allen, Mr. William Henry) because:
> * The text field `Sex` does not contain [female]
> * The numeric field `Age` is between [34.0] and [37.0]
> * The numeric field `Pclass` is greater than [2.5]

---

### 🛠 Installation
To install the package in editable mode for local development:
```bash
pip install -e .
```

## 📋 Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Function Documentation](#function-documentation)
- [Supported Problem Types](#supported-problem-types)
- [Examples](#examples)

## 🚀 Installation

```python
from algorithmeai import Snake
```

No external dependencies required - uses only Python standard library.

## 📖 Overview

**Snake** is a multiclass CSV data handler that builds explanatory models based on logical clauses. The system identifies "lookalikes" (similar data points) and generates interpretable rules for classification.

### Key Features

- **Explainable Classification**: Generates understandable logical rules
- **Multi-type Support**: Handles binary, integer, float, and text data
- **Complete Audit**: Provides detailed explanations for each prediction
- **Validation**: Pruning process to optimize the model

## ⚡ Quick Start

```python
# Create a Snake model
model = Snake(
    csv_path="data.csv",
    n_layers=100,
    vocal=True,
    target_index=0,
    excluded_features_index=[1, 2]
)

# Make a prediction
datapoint = {"feature1": 5.2, "feature2": "text", ...}
prediction = model.get_prediction(datapoint)

# Get a complete explanation
audit = model.get_audit(datapoint)

# Save the model
model.to_json("my_model.json")

# Load an existing model
model = Snake("my_model.json")
```

## 📚 Function Documentation

### Initialization

#### `Snake(csv_path, n_layers=100, vocal=True, target_index=0, excluded_features_index=[])`

Creates a Snake instance from a CSV or JSON file.

**Parameters:**
- `csv_path` (str): Path to CSV or JSON file
- `n_layers` (int): Number of logical layers to build (default: 100)
- `vocal` (bool): Enable verbose logging (default: True)
- `target_index` (int): Index of the target column (default: 0)
- `excluded_features_index` (list): Indices of columns to exclude from training

**Example:**
```python
model = Snake("dataset.csv", n_layers=50, target_index=0)
```

---

### Prediction Functions

#### `get_prediction(X)`

Predicts the most probable class for a data point.

**Parameters:**
- `X` (dict): Dictionary with feature values

**Returns:** Predicted target value

**Example:**
```python
prediction = model.get_prediction({"age": 25, "name": "Alice"})
```

---

#### `get_probability(X)`

Computes the probability vector for all classes.

**Parameters:**
- `X` (dict): Dictionary with feature values

**Returns:** Dictionary {class: probability}

**Example:**
```python
probas = model.get_probability({"age": 25, "name": "Alice"})
# Result: {0: 0.75, 1: 0.25}
```

---

#### `get_lookalikes(X)`

Identifies similar data points in the training set.

**Parameters:**
- `X` (dict): Dictionary with feature values

**Returns:** List of triplets [index, class, conditions]

**Example:**
```python
lookalikes = model.get_lookalikes({"age": 25, "name": "Alice"})
# Result: [[42, 1, [0, 5, 12]], [87, 1, [3, 7]]]
```

---

### Audit and Explanation Functions

#### `get_audit(X)`

Generates a complete audit report for a prediction.

**Parameters:**
- `X` (dict): Dictionary with feature values

**Returns:** String with detailed audit

**Example:**
```python
audit = model.get_audit({"age": 25, "name": "Alice"})
print(audit)
```

---

#### `get_plain_text_assertion(condition, l)`

Converts a logical condition into readable text.

**Parameters:**
- `condition` (list): List of clause indices
- `l` (int): Index of the lookalike data point

**Returns:** Textual description of the condition

---

#### `get_augmented(X)`

Enriches a data point with all available information.

**Parameters:**
- `X` (dict): Dictionary with feature values

**Returns:** Enriched dictionary with lookalikes, probabilities, prediction, and audit

**Example:**
```python
augmented = model.get_augmented({"age": 25, "name": "Alice"})
# Contains: X + Lookalikes, Probability, Prediction, Audit
```

---

### Internal Logical Functions

#### `apply_literal(X, literal)`

Tests if a data point satisfies a logical literal.

**Parameters:**
- `X` (dict): Data point
- `literal` (list): [index, value, negation, data_type]

**Returns:** True if the literal is satisfied, False otherwise

---

#### `apply_clause(X, clause)`

Tests if a data point satisfies a clause (OR of literals).

**Parameters:**
- `X` (dict): Data point
- `clause` (list): List of literals

**Returns:** True if at least one literal is satisfied

---

### Model Management

#### `to_json(fout="snakeclassifier.json")`

Saves the model to JSON format.

**Parameters:**
- `fout` (str): Output file path

**Example:**
```python
model.to_json("my_model_v1.json")
```

---

#### `from_json(filepath="snakeclassifier.json")`

Loads a model from a JSON file.

**Parameters:**
- `filepath` (str): Path to the file to load

---

#### `make_validation(Xs, pruning_coef=0.5)`

Validates and prunes the model on a validation set.

**Parameters:**
- `Xs` (list): List of validation data points
- `pruning_coef` (float): Pruning coefficient (0.5 = reduce by half)

**Example:**
```python
validation_set = [{"age": 30, "name": "Bob", "target": 1}, ...]
model.make_validation(validation_set, pruning_coef=0.6)
```

---

### Utilities

#### `read_csv(fname)`

Parses a CSV file with quote handling.

**Parameters:**
- `fname` (str): CSV file path

**Returns:** Tuple (header, data)

---

#### `make_population(fname, drop=False)`

Creates the data population from a CSV.

**Parameters:**
- `fname` (str): CSV file path
- `drop` (bool): If True, removes duplicates

**Returns:** List of dictionaries

---

## 🎯 Supported Problem Types

Snake automatically detects the problem type:

### 1. **Binary (0/1 or True/False)**
```python
# Examples: fraud/not fraud, sick/healthy
```

### 2. **Multiclass Integers**
```python
# Examples: categories 0, 1, 2, 3
```

### 3. **Regression (Floating Point Numbers)**
```python
# Examples: prices, temperatures, scores
```

### 4. **Text Classification**
```python
# Examples: named categories, text labels
```

## 💡 Complete Examples

### Example 1: Simple Classification

```python
# Create and train the model
model = Snake("iris.csv", n_layers=50, target_index=4)

# Predict
new_flower = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

prediction = model.get_prediction(new_flower)
probabilities = model.get_probability(new_flower)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
```

### Example 2: Detailed Audit

```python
# Get a complete explanation
audit_report = model.get_audit(new_flower)
print(audit_report)

# Output:
# ### BEGIN AUDIT ###
# ### Datapoint {...}
# ## Number of lookalikes 15
# ## Predicted outcome (max proba) [setosa]
# 
# # Probability of being equal to class setosa : 93.3%
# # Probability of being equal to class versicolor : 6.7%
# ...
```

### Example 3: Validation and Optimization

```python
# Load a validation set
validation_data = [
    {"feature1": 1.2, "feature2": "A", "target": 0},
    {"feature1": 3.4, "feature2": "B", "target": 1},
    # ...
]

# Prune the model
model.make_validation(validation_data, pruning_coef=0.7)

# Save the optimized model
model.to_json("model_optimized.json")
```

## 📊 Data Structure

### Input CSV Format
```csv
target,feature1,feature2,feature3
1,5.2,"text A",100
0,3.1,"text B",200
```

### Output JSON Format
The saved model contains:
- `population`: Training data
- `header`: Column names
- `clauses`: Learned logical rules
- `lookalikes`: Indices of similar points
- `datatypes`: Type of each column
- `log`: Operation history

## ⚠️ Important Notes

- CSV files must be formatted by pandas (`df.to_csv()`)
- Triple quotes in CSV are handled automatically
- Missing values are converted to 0 (numeric) or "" (text)
- Complexity is O(n·log(n)·m·bucket²) where n = samples, m = features, bucket = bucket size

## 📝 License

© Charles Dana, December 2025

This project is licensed under the MIT License.





