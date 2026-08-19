# 🌊 Aquanga

### Predictive Water Monitoring & Early Warning System

Aquanga is an end-to-end machine learning and deep learning system for forecasting water-quality trends and identifying potential deterioration risks in the Ganga River.

Instead of only analyzing the current condition of water, Aquanga uses historical water-quality and environmental data to forecast future conditions and provide an early-warning mechanism for potentially high-risk locations.

---

# 🎯 Problem

Water-quality monitoring systems provide valuable information about the current and historical state of rivers.

However, detecting deterioration only after it occurs can limit the time available for investigation and preventive action.

Aquanga addresses this challenge by combining:

* Time-series forecasting
* Machine Learning
* Deep Learning
* Environmental data
* Risk assessment
* REST APIs
* Database engineering
* Interactive visualization

to build a predictive monitoring system.

---

# 💡 Solution

Aquanga follows an end-to-end predictive monitoring pipeline.

### 🔄 Overall Project Flow

```text
┌──────────────────────────────┐
│   Water + Environmental Data │
│   Historical / Current Data  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      Data Preprocessing      │
│ Cleaning • Missing Values    │
│ Scaling • Time Alignment     │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     Feature Engineering      │
│ Temporal + Environmental     │
│ Features                     │
└──────────────┬───────────────┘
               ↓
       ┌───────┴────────┐
       ↓                ↓
┌──────────────┐  ┌──────────────┐
│ ML Baselines │  │ Deep Learning│
│ RF / XGBoost │  │ CNN / LSTM   │
└──────┬───────┘  └──────┬───────┘
       └───────┬─────────┘
               ↓
       Model Comparison
               ↓
      Best Model Selection
               ↓
       Future Forecasting
               ↓
       Risk Assessment
               ↓
        ┌──────┴───────┐
        ↓              ↓
   Dashboard      Early Warning
        │              │
        └──────┬───────┘
               ↓
        Aquanga Copilot
```

> **Core idea:** Move from monitoring current water conditions to forecasting future water-quality risks.

---

# 🚀 Key Features

## 1. Water-Quality Forecasting

Aquanga forecasts future water-quality conditions using historical observations.

Potential features include:

* pH
* Dissolved Oxygen (DO)
* Biological Oxygen Demand (BOD)
* Chemical Oxygen Demand (COD)
* Temperature
* Turbidity
* Conductivity
* Rainfall
* Flow-related measurements
* Location
* Timestamp

The exact feature set will depend on the availability and quality of the dataset.

---

## 2. Time-Series Modeling

Water quality changes over time, making temporal patterns important.

Aquanga experiments with:

* Traditional Machine Learning
* 1D CNN
* LSTM
* CNN + LSTM

The objective is to determine which approach provides the best forecasting performance on the available data.

---

# 🧠 Model Architecture

The primary deep-learning candidate is a hybrid **CNN-LSTM** architecture.

### CNN-LSTM Flow

```text
┌──────────────────────────┐
│ Historical Time-Series   │
│ pH • DO • BOD • COD • ... │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      Input Sequence      │
│       Past N Days        │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│          1D CNN          │
│  Local Temporal Patterns │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│     Feature Maps         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│          LSTM            │
│ Temporal Dependencies    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      Dense Layer         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Future Prediction     │
│      24 / 48 / 72h       │
└──────────────────────────┘
```

### Why 1D CNN?

1D CNN can learn local patterns in sequential water-quality observations.

### Why LSTM?

LSTM can capture dependencies across previous time steps.

### Why CNN + LSTM?

The hybrid architecture attempts to combine:

```text
CNN  → Local / short-term patterns

LSTM → Temporal / long-term dependencies
```

The architecture will be validated experimentally rather than assuming that a more complex model is always better.

---

# 📊 Model Benchmarking

Aquanga follows a **baseline-first approach**.

Models will be evaluated progressively:

| Model             | Role                       |
| ----------------- | -------------------------- |
| Linear Regression | Simple baseline            |
| Random Forest     | Non-linear baseline        |
| XGBoost           | Strong tree-based baseline |
| 1D CNN            | Deep-learning baseline     |
| LSTM              | Time-series deep learning  |
| CNN + LSTM        | Hybrid deep-learning model |

This allows the project to answer an important engineering question:

> **Does the more complex deep-learning architecture actually improve forecasting performance?**

---

# 📏 Evaluation Metrics

For regression-based forecasting:

* MAE
* RMSE
* R²

For risk classification, if implemented:

* Precision
* Recall
* F1-score
* Confusion Matrix

Time-series evaluation will use chronological train/validation/test splits to reduce the risk of data leakage.

---

# 🚨 Early Warning System

The forecasting model feeds into a risk-assessment layer.

```text
┌────────────────────┐
│ Future Forecast    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Risk Assessment    │
│ / Threshold Logic  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Risk Classification│
└─────────┬──────────┘
          ↓
   ┌──────┼──────┐
   ↓      ↓      ↓
  LOW   MEDIUM   HIGH
   🟢     🟡      🔴
          ↓
┌────────────────────┐
│   Early Warning    │
└────────────────────┘
```

Possible risk levels:

```text
🟢 LOW
🟡 MEDIUM
🔴 HIGH
```

Example:

```text
Station: Varanasi

Current Risk: Medium

Forecast Horizon: 48 hours

Predicted Risk: High
```

The system is intended to help prioritize further monitoring and investigation. It does not replace official environmental monitoring or regulatory decisions.

---

# 🗺️ Monitoring Dashboard

The dashboard provides an interactive view of:

* Monitoring stations
* Current measurements
* Historical trends
* Forecasts
* Risk levels
* Water-quality parameters
* Location-based information
* Early-warning indicators

Example:

```text
Station      Current       Forecast
------------------------------------
Station A    🟢 Low        🟢 Low
Station B    🟡 Medium     🟡 Medium
Station C    🟢 Low        🔴 High
Station D    🟡 Medium     🔴 High
```

---

# 🤖 Aquanga Copilot

Aquanga Copilot is an optional natural-language interface for explaining model predictions.

Instead of presenting only numerical values, the assistant can convert model outputs into understandable explanations.

Example:

```text
DO: 5.8 mg/L
BOD: 4.2 mg/L
Turbidity: Increasing

Aquanga Copilot:

"The selected station shows a deteriorating trend.
The forecasting model estimates a higher water-quality
risk over the next 48 hours."
```

The assistant is designed to explain model outputs rather than make independent scientific or regulatory decisions.

---

# 🏗️ System Architecture

### Production Architecture

```text
                     DATA SOURCES
                          │
                          ▼
              ┌──────────────────────┐
              │ CSV / APIs / Sensors │
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │    Data Pipeline     │
              │ Pandas + NumPy       │
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │ Feature Engineering  │
              └──────────┬───────────┘
                         ↓
               ┌─────────┴─────────┐
               ↓                   ↓
       ┌───────────────┐   ┌───────────────┐
       │ Traditional ML│   │ Deep Learning │
       │ RF / XGBoost  │   │ CNN + LSTM    │
       └───────┬───────┘   └───────┬───────┘
               └─────────┬─────────┘
                         ↓
                ┌────────────────┐
                │ Model Selection│
                └───────┬────────┘
                        ↓
                ┌────────────────┐
                │   Forecasting  │
                └───────┬────────┘
                        ↓
                ┌────────────────┐
                │  Risk Engine   │
                └───────┬────────┘
                        ↓
                ┌────────────────┐
                │    FastAPI     │
                │    REST API    │
                └───────┬────────┘
                        ↓
              ┌─────────┴─────────┐
              ↓                   ↓
      ┌──────────────┐     ┌──────────────┐
      │ PostgreSQL   │     │  Dashboard   │
      │              │     │  Streamlit   │
      └──────────────┘     └───────┬──────┘
                                   ↓
                           Early Warnings
                                   ↓
                           Aquanga Copilot
```

---

# 🛠️ Tech Stack

## Programming

* Python

## Data Science

* Pandas
* NumPy
* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost

## Deep Learning

* TensorFlow
* Keras
* 1D CNN
* LSTM

## Backend

* FastAPI
* SQLAlchemy
* Pydantic

## Database

* PostgreSQL

## Dashboard

* Streamlit
* Folium / PyDeck

## DevOps

* Git
* GitHub
* Docker
* Docker Compose

---

# 📁 Project Structure

```text
Aquanga/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_experiments.ipynb
│
├── ml/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── create_sequences.py
│   ├── train_baseline.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── train_cnn.py
│   ├── train_cnn_lstm.py
│   └── evaluate.py
│
├── models/
│   ├── baseline/
│   ├── xgboost/
│   ├── lstm/
│   └── cnn_lstm/
│
├── app/
│   ├── main.py
│   │
│   ├── routers/
│   │   ├── prediction.py
│   │   ├── stations.py
│   │   └── alerts.py
│   │
│   ├── schemas/
│   │   ├── prediction.py
│   │   └── station.py
│   │
│   ├── services/
│   │   ├── prediction_service.py
│   │   ├── risk_service.py
│   │   └── alert_service.py
│   │
│   └── database/
│       ├── database.py
│       ├── models.py
│       └── crud.py
│
├── dashboard/
│   ├── app.py
│   ├── components/
│   └── pages/
│
├── docs/
│   └── architecture/
│       ├── system-flow.png
│       ├── cnn-lstm.png
│       └── production-architecture.png
│
├── tests/
│   ├── test_prediction.py
│   ├── test_api.py
│   └── test_risk.py
│
├── scripts/
│   ├── train.py
│   └── seed_database.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 🔄 Development Workflow

```text
1. Data Collection
       ↓
2. Data Understanding
       ↓
3. Data Cleaning
       ↓
4. Exploratory Data Analysis
       ↓
5. Feature Engineering
       ↓
6. Time-Series Sequence Creation
       ↓
7. Baseline Models
       ↓
8. XGBoost
       ↓
9. LSTM
       ↓
10. CNN
       ↓
11. CNN + LSTM
       ↓
12. Model Evaluation
       ↓
13. Select Best Model
       ↓
14. FastAPI Integration
       ↓
15. PostgreSQL Integration
       ↓
16. Dashboard
       ↓
17. Early-Warning System
       ↓
18. Dockerization
       ↓
19. Deployment
```

---

# 🧪 Engineering Practices

Aquanga follows production-oriented practices including:

* Modular project architecture
* Train/validation/test separation
* Time-aware data splitting
* Reproducible preprocessing
* Model versioning
* API validation
* Database persistence
* Automated testing
* Environment-variable based configuration
* Containerized deployment

---

# 🔐 Configuration

Sensitive configuration should not be committed to GitHub.

Example environment variables:

```env
DATABASE_URL=
MODEL_PATH=
API_KEY=
```

Use `.env` locally and maintain `.env.example` for documentation.

---

# 📈 Future Scope

Possible future improvements:

* Real-time IoT sensor integration
* Weather API integration
* Satellite imagery
* River-flow data
* Transformer-based forecasting
* Explainable AI
* Automated notifications
* Model monitoring
* Cloud deployment
* Multi-river expansion
* Advanced geospatial analytics

---

# 🎯 Project Objective

Aquanga demonstrates how machine learning, deep learning, time-series forecasting, backend engineering, databases, and deployment can be combined to build a real-world environmental decision-support system.

The core objective is:

> **From monitoring current conditions to forecasting future water-quality risk.**

---

# 👩‍💻 Status

🚧 **Currently under development**

### Planned Milestones

* [ ] Dataset collection
* [ ] Data cleaning
* [ ] Exploratory analysis
* [ ] Feature engineering
* [ ] Baseline ML model
* [ ] XGBoost model
* [ ] LSTM model
* [ ] CNN model
* [ ] CNN-LSTM model
* [ ] Model comparison
* [ ] FastAPI backend
* [ ] PostgreSQL integration
* [ ] Streamlit dashboard
* [ ] Early-warning system
* [ ] Aquanga Copilot
* [ ] Dockerization
* [ ] Deployment

---

# 📜 License

This project is intended for educational, research, and hackathon purposes.
