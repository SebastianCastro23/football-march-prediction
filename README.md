# ⚽ Football Match Outcome Prediction
> A machine learning project to predict the outcome of European football matches (Home Win / Draw / Away Win)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-1.5-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Predicting football match outcomes is one of the most challenging problems in sports analytics. Unlike many classification tasks, football has high inherent randomness — a single moment or refereeing decision can change everything.

This project explores whether machine learning can beat simple baselines using historical match data, team tactical attributes, and betting odds from the [European Soccer Database](https://www.kaggle.com/hugomathien/soccer).

---

## 🎯 Results

| Model | Accuracy |
|---|---|
| Random Guess *(baseline)* | 33.3% |
| Always Predict Home Win *(naive baseline)* | ~46.0% |
| Logistic Regression | 47.5% |
| HistGradientBoosting | 47.1% |
| XGBoost | 49.8% |
| **Random Forest** ✅ | **50.4%** |

> 💡 Professional betting companies achieve ~55-60%. The remaining gap is largely due to football's inherent unpredictability.

---

## 📊 Key Findings

- **Home advantage is real** — 45.9% of all matches are won by the home team, making it the single strongest predictor.
- **Betting odds are the most predictive feature** — Converting Bet365 odds to implied probabilities added ~3-4% accuracy across all models.
- **Draws are nearly impossible to predict** — Even with class balancing techniques, Draw F1-score remained below 0.35, consistent with academic literature.
- **Gradient boosting beats deep learning on tabular data** — HistGradientBoosting consistently outperformed a 4-layer neural network on this dataset size (~25k rows).

---

## 🗂️ Dataset

**Source:** [European Soccer Database — Kaggle](https://www.kaggle.com/hugomathien/soccer)

| Table | Records | Description |
|---|---|---|
| Match | 25,979 | Match results across 11 leagues (2008–2016) |
| Player | 11,060 | Player info and FIFA attributes |
| Team | 299 | Team names and tactical attributes |
| Team_Attributes | 1,458 | FIFA tactical ratings per season |
| Player_Attributes | 183,978 | FIFA player ratings over time |

---

## ⚙️ Feature Engineering

All features are computed using only **pre-match information** to avoid data leakage.

| Feature | Description |
|---|---|
| `home_form` / `away_form` | Points earned in last 5 matches |
| `form_diff` | Difference in form between home and away team |
| `h2h_home_rate` | Home team win rate in last 5 head-to-head matches |
| `home_goals_avg` / `away_goals_avg` | Average goals scored in last 5 matches |
| `home/away_buildUpPlaySpeed` | FIFA tactical attribute — build-up speed |
| `home/away_defencePressure` | FIFA tactical attribute — defensive pressure |
| `home/away_defenceAggression` | FIFA tactical attribute — defensive aggression |
| `imp_prob_H/D/A` | Bet365 implied probabilities (1 / decimal odd) |

> ⚠️ In-game statistics (shots, possession, cards) were **excluded** as they would not be available before the match starts.

---

## 🧠 Models Compared

1. **Logistic Regression** — linear baseline
2. **Random Forest** — ensemble of decision trees *(best)*
3. **HistGradientBoosting** — sklearn's fast gradient boosting
4. **XGBoost** — industry-standard gradient boosting
5. **Deep Neural Network** — 4-layer Keras model with BatchNormalization and Dropout

---

## 🔧 How to Run

### Option 1 — Google Colab *(recommended)*
1. Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Add your Kaggle credentials in cell 2
3. Run all cells top to bottom

### Option 2 — Local
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/football-match-prediction.git
cd football-match-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebook.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
kaggle
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
football-match-prediction/
│
├── notebook.ipynb          # Main project notebook
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── images/                 # Plot screenshots
    ├── outcome_distribution.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 🚀 Future Work

- Include aggregated player FIFA ratings as team strength features
- Add ELO ratings as a dynamic team strength metric
- Incorporate more seasons for better generalization
- Explore ensemble stacking of multiple models
- Predict exact match score (regression) rather than just outcome

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Your Name**
- GitHub: [@yourhandle](https://github.com/SebastianCastro23)
- LinkedIn: [linkedin.com/in/yourprofile](https://www.linkedin.com/in/juan-sebastian-castro-pardo-62a2a5233/)
