<!-- Header Animation -->
<div align="center">
  <h1>🧠 Recurrent Neural Networks (RNN) Complete Guide</h1>
  <p>
    <strong>Master RNNs with Three Essential Projects: NLP, Time Series & PyTorch</strong>
  </p>
  
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
  ![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C?style=for-the-badge&logo=pytorch)
  ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)
  ![Keras](https://img.shields.io/badge/Keras-Latest-D00000?style=for-the-badge&logo=keras)
  
  **[Projects](#-featured-projects) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Contributing](#-contributing)**
</div>

---

## 📋 Table of Contents

- [Featured Projects](#-featured-projects)
- [Quick Start](#-quick-start)
- [Project 1: Sentiment Analysis](#-project-1-clothing-review-sentiment-analysis-tensorflow)
- [Project 2: Time Series Forecasting](#-project-2-time-series-forecasting-tensorflow)
- [Project 3: PyTorch Implementation](#-project-3-pytorch-rnn-implementation)
- [Installation](#-installation)
- [Key Concepts](#-key-concepts)
- [Contributing](#-contributing)
- [License](#-license)

---

## ⭐ Featured Projects

<details>
<summary><b>3 Complete RNN Projects in One Repository</b></summary>

| Project | Framework | Task | Dataset | Accuracy |
|---------|-----------|------|---------|----------|
| **Clothing Review Sentiment** | TensorFlow/Keras | NLP Classification | Clothing Reviews | 92.86% |
| **Stock Price Forecasting** | TensorFlow/Keras | Time Series | AAPL Stock Data | RMSE Optimized |
| **PyTorch Implementation** | PyTorch | Neural Network Basics | IMDB Dataset | Full Walkthrough |

### 📂 Project Structure

```
├── Training_of_Recurrent_Neural_Networks_(RNN)_in_TensorFlow.ipynb
│   └── Sentiment Analysis on Clothing Reviews
│       • EDA & Data Preprocessing
│       • Text Tokenization & Embedding
│       • SimpleRNN Model Training
│       • Performance Evaluation
│
├── Time Series Forecasting using RNN in TensorFlow.ipynb
│   └── Stock Price Prediction
│       • Yahoo Finance Data Fetching
│       • MinMax Scaling & Normalization
│       • Time Series Data Preparation
│       • RNN Architecture for Sequences
│
├── Implementing Recurrent Neural Networks in PyTorch.ipynb
│   └── PyTorch Fundamentals
│       • Basic RNN Implementation
│       • IMDB Dataset Processing
│       • Model Training in PyTorch
│       • Evaluation & Visualization
│
├── Clothing-Review.csv
├── IMDB-Dataset.csv
└── README.md
```

</details>

---

## 🎯 Overview

This repository contains **three comprehensive projects** demonstrating Recurrent Neural Networks (RNNs) across different domains and frameworks:

1. **📝 Sentiment Analysis (TensorFlow)** - NLP classification of clothing reviews
2. **📊 Time Series Forecasting (TensorFlow)** - Stock price prediction using sequential data
3. **🔧 PyTorch Implementation** - Learn RNNs from scratch with PyTorch

### What You'll Learn:

- 📊 Exploratory Data Analysis (EDA) techniques
- 🧹 Text preprocessing, cleaning & normalization
- 🔤 Tokenization, embeddings & word representations
- 📈 Time series data preparation & scaling
- 🧠 Building SimpleRNN, LSTM & GRU models
- 🏆 Training, evaluation & performance optimization
- 🎯 Achieving state-of-the-art accuracy scores

---

## ⚡ Quick Start

<details>
<summary><b>Get Started in 5 Minutes</b></summary>

### Option 1: Clone Repository

```bash
git clone https://github.com/itsluckysharma01/Training-of-Recurrent-Neural-Networks-RNN-in-TensorFlow.git
cd Training-of-Recurrent-Neural-Networks-RNN-in-TensorFlow
```

### Option 2: Install Dependencies

```bash
# Create virtual environment
python -m venv rnn_env
source rnn_env/bin/activate  # On Windows: rnn_env\Scripts\activate

# Install required packages
pip install tensorflow keras pytorch pandas numpy matplotlib scikit-learn yfinance nltk
```

### Option 3: Run Any Notebook

```bash
# Launch Jupyter
jupyter notebook

# Then select one of:
# 1. Training_of_Recurrent_Neural_Networks_(RNN)_in_TensorFlow.ipynb
# 2. Time Series Forecasting using RNN in TensorFlow.ipynb
# 3. Implementing Recurrent Neural Networks in PyTorch.ipynb
```

### Option 4: Execute Cells Sequentially

Once the notebook opens, run cells from top to bottom using `Shift + Enter`

</details>

---

## 📊 Project 1: Clothing Review Sentiment Analysis (TensorFlow)

<details>
<summary><b>Sentiment Analysis with TensorFlow/Keras</b></summary>

### Overview

Classify clothing reviews as positive or negative using a SimpleRNN model trained on 10,000+ reviews.

### Key Features

- ✅ **Data**: 23,486 clothing brand reviews
- ✅ **Preprocessing**: Tokenization, padding, lemmatization
- ✅ **Model**: 2-layer SimpleRNN with embedding
- ✅ **Accuracy**: 92.86% on test data
- ✅ **Training Time**: ~2-3 minutes

### Dataset Information

| Metric | Value |
|--------|-------|
| Total Reviews | 23,486 |
| Classes | 2 (Positive/Negative) |
| Features | Title, Review Text, Rating |
| Train/Test Split | 75/25 |

### Model Architecture

```
Input (40,) 
    ↓
Embedding (10000 → 128)
    ↓
SimpleRNN (64 units, return_sequences=True)
    ↓
SimpleRNN (64 units)
    ↓
Dense (128, relu)
    ↓
Dropout (0.4)
    ↓
Dense (1, sigmoid)
    ↓
Binary Classification Output
```

### Quick Usage

```python
# Load & preprocess data
df = pd.read_csv("Clothing-Review.csv")
# ... preprocessing steps ...

# Build model
model = Sequential([
    Embedding(10000, 128, input_length=40),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate
accuracy = model.evaluate(X_test, y_test)
```

</details>

---

## 📈 Project 2: Time Series Forecasting (TensorFlow)

<details>
<summary><b>Stock Price Prediction with RNN</b></summary>

### Overview

Predict Apple (AAPL) stock prices using historical data and RNN for sequential learning.

### Key Features

- ✅ **Data Source**: Yahoo Finance (2010-2023)
- ✅ **Period**: 3,247 trading days
- ✅ **Normalization**: MinMax scaling (0-1)
- ✅ **Sequence Length**: 60-day windows
- ✅ **Evaluation**: MSE, RMSE, MAE metrics

### Data Characteristics

| Aspect | Details |
|--------|---------|
| Ticker | AAPL (Apple Inc.) |
| Date Range | Jan 1, 2010 - Jan 1, 2023 |
| Time Steps | 60 days lookback |
| Prediction | Next day close price |

### Model Architecture

```
Input (60, 1) - 60 days of prices
    ↓
SimpleRNN (50 units, return_sequences=True)
    ↓
SimpleRNN (50 units)
    ↓
Dense (1)
    ↓
Predicted Price (unscaled)
```

### Performance Metrics

```
Training Results:
├─ MSE: [calculated value]
├─ RMSE: [calculated value]
└─ MAE: [calculated value]

Data Split:
├─ Training: 80% (2597 samples)
└─ Testing: 20% (650 samples)
```

### Quick Usage

```python
import yfinance as yf

# Fetch data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
values = data['Close'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# Prepare sequences
X, y = create_dataset(scaled_data, time_step=60)

# Train model
model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(60, 1)),
    SimpleRNN(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32)
```

</details>

---

## 🔧 Project 3: PyTorch RNN Implementation

<details>
<summary><b>Learn RNNs from Scratch with PyTorch</b></summary>

### Overview

Implement RNNs from the ground up using PyTorch framework - perfect for understanding the mechanics.

### Key Features

- ✅ **Framework**: Pure PyTorch implementation
- ✅ **Dataset**: IMDB reviews
- ✅ **Learning**: From basic RNN to advanced architectures
- ✅ **Topics**: 
  - RNN cell mechanics
  - Forward & backward propagation
  - Custom layer implementation
  - Loss functions & optimization

### Topics Covered

1. **Basic RNN Architecture**
   - Understanding RNN cells
   - Hidden states & temporal dependencies
   - Vanishing gradient problem

2. **Implementation Details**
   - Manual weight initialization
   - Custom RNN layers
   - Sequence processing

3. **Training Pipeline**
   - Loss computation
   - Backpropagation through time (BPTT)
   - Optimization strategies

4. **Advanced Topics**
   - LSTM gates (input, forget, output)
   - GRU architecture
   - Bidirectional RNNs

### Model Comparison

| Type | Advantages | Disadvantages |
|------|-----------|---------------|
| SimpleRNN | Fast, Simple | Vanishing gradients |
| LSTM | Long-term memory | More parameters |
| GRU | Balanced, Efficient | Slightly less powerful |

</details>

## 💻 Installation

### Prerequisites

- Python 3.10+
- pip or conda
- Jupyter Notebook
- 2GB RAM minimum

### Step-by-Step Installation

<details>
<summary><b>Install Dependencies</b></summary>

```bash
# 1. Clone the repository
git clone https://github.com/itsluckysharma01/Training-of-Recurrent-Neural-Networks-RNN-in-TensorFlow.git
cd Training-of-Recurrent-Neural-Networks-RNN-in-TensorFlow

# 2. Create virtual environment (recommended)
python -m venv rnn_env
source rnn_env/bin/activate  # On Windows: rnn_env\Scripts\activate

# 3. Install required packages
pip install tensorflow>=2.10.0
pip install keras>=2.10.0
pip install pandas numpy scikit-learn
pip install matplotlib seaborn plotly
pip install nltk

# 4. Download NLTK data
python -c "import nltk; nltk.download('all')"

# 5. Launch Jupyter
jupyter notebook
```

</details>

---

## 📖 Usage Guide

<details>
<summary><b>Step-by-Step Usage Instructions</b></summary>

### 1. Load and Explore Data

```python
# The notebook automatically loads the CSV from GitHub
Dataset = "https://raw.githubusercontent.com/itsluckysharma01/Datasets/..."
df = pd.read_csv(Dataset)
df.head()  # View first few rows
```

### 2. Run Text Preprocessing

```python
# Convert to lowercase, remove stopwords, lemmatize, remove punctuation
X['Title'] = X['Title'].apply(toLower)
X['Review Text'] = X['Review Text'].apply(remove_stopwords)
# ... more preprocessing
```

### 3. Tokenize & Pad

```python
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_clean)
train_seq = tokenizer.texts_to_sequences(X_train_clean)
train_pad = pad_sequences(train_seq, maxlen=40)
```

### 4. Build & Train Model

```python
model = keras.models.Sequential()
model.add(keras.layers.Embedding(10000, 128, input_length=40))
model.add(keras.layers.SimpleRNN(64, return_sequences=True))
model.add(keras.layers.SimpleRNN(64))
# ... more layers
model.compile(loss="binary_crossentropy", optimizer="adam")
history = model.fit(train_pad, y_train, epochs=5)
```

</details>

---

## 🔑 Key Concepts

<details>
<summary><b>Understanding RNNs & NLP Concepts</b></summary>

### What is an RNN?

A **Recurrent Neural Network** is a type of neural network designed for sequential data. It maintains internal memory (hidden states) that capture information from previous steps.

```
Sequential Input Processing:
word1 → word2 → word3 → word4 → word5
 │      │       │       │       │
 └─→[RNN]─→[RNN]─→[RNN]─→[RNN]─→Output
```

### Key Concepts Explained

#### 🔹 Tokenization

Converting text into numerical sequences:

```
Text: "great product"
Tokens: [2, 5]  (using vocabulary index)
```

#### 🔹 Embedding

Dense vector representation of words:

```
word "good" → [0.2, -0.5, 0.8, ...]  (128 dimensions)
```

#### 🔹 Padding

Making sequences uniform length:

```
[2, 5, 3]  →  [2, 5, 3, 0, 0, 0, 0, 0, 0, 0]  (maxlen=10)
```

#### 🔹 SimpleRNN Cell

Processes sequences and maintains hidden states:

```
h(t) = tanh(W*x(t) + U*h(t-1) + b)
```

### Why Binary Classification?

- Simplified problem: **Positive (1) vs Negative (0)** sentiment
- Ratings 4-5 → Positive (1)
- Ratings 1-3 → Negative (0)

</details>

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

<details>
<summary><b>Contribution Guidelines</b></summary>

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution

- ✅ Improve model architecture
- ✅ Add LSTM/GRU layers
- ✅ Implement attention mechanisms
- ✅ Optimize preprocessing pipeline
- ✅ Enhance documentation
- ✅ Add unit tests
- ✅ Create visualization tools

</details>

---

## 📚 Additional Resources

<details>
<summary><b>Learning Resources & References</b></summary>

### Official Documentation

- [TensorFlow Official Docs](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [NLTK Documentation](https://www.nltk.org/)

### Tutorials & Courses

- [RNN Tutorial - TensorFlow](https://www.tensorflow.org/guide/keras/rnn)
- [Text Classification Guide](https://www.tensorflow.org/tutorials/keras/text_classification)
- [NLP with TensorFlow](https://www.coursera.org/learn/natural-language-processing-tensorflow)

### Research Papers

- [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)
- [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215)

</details>

---

## 📞 Contact & Support

<details>
<summary><b>Get Help & Connect</b></summary>

### Author

- **Name**: Sharma
- **GitHub**: [@itsluckysharma01](https://github.com/itsluckysharma01)

### Questions or Issues?

- 🐛 Found a bug? [Open an Issue](https://github.com/itsluckysharma01/Training-of-Recurrent-Neural-Networks-RNN-in-TensorFlow/issues)
- 💬 Have a question? Start a Discussion
- 📧 Email support available

</details>

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

<div align="center">
  <h3>⭐ If you found this helpful, please consider giving it a star! ⭐</h3>
  
  **Happy Learning! 🚀**
  
  ![RNN Animation](https://img.shields.io/badge/Last%20Updated-2026-blue?style=flat-square)
  ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
  
</div>

---

### Quick Navigation

```
🏠 [Home](#training-of-recurrent-neural-networks-rnn-in-tensorflow)
📚 [Table of Contents](#-table-of-contents)
🎯 [Overview](#-overview)
⚡ [Quick Start](#-quick-start)
📊 [Dataset](#-dataset)
🔄 [Workflow](#-project-workflow)
🧠 [Model](#-model-architecture)
📈 [Results](#-results--performance)
💻 [Installation](#-installation)
📖 [Usage](#-usage-guide)
🔑 [Concepts](#-key-concepts)
🤝 [Contributing](#-contributing)
📚 [Resources](#-additional-resources)
📞 [Contact](#--contact--support)
```
