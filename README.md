# Heart Disease Prediction

A machine-learning project that predicts whether a patient has **heart disease** from 13 medical measurements, using **Logistic Regression**. It was built as a Month 1 task of an AI/ML internship, and it includes a step many quick projects skip: the **duplicate rows in the dataset are removed before the data is split**, so the score is not inflated.

> **Educational project only.** It is not a medical device and must not be used to diagnose or treat anyone.

## The data

`heart.csv` has 1,025 rows with 13 inputs and a `target` column (1 = heart disease, 0 = no disease).

| Column | Meaning (standard UCI heart-disease coding) |
|---|---|
| `age`, `sex` | Age in years; 0 = female, 1 = male |
| `cp` | Chest pain type (0 to 3) |
| `trestbps`, `chol` | Resting blood pressure; serum cholesterol |
| `fbs` | Fasting blood sugar above 120 mg/dl (0 or 1) |
| `restecg` | Resting ECG result (0 to 2) |
| `thalach` | Maximum heart rate reached |
| `exang` | Exercise-induced angina (0 or 1) |
| `oldpeak`, `slope` | ST depression during exercise and the slope of the peak ST segment |
| `ca`, `thal` | Major vessels coloured by fluoroscopy; thalassemia test code |

**Only 302 of the 1,025 rows are unique.** The other 723 are exact duplicates, so the notebook drops them first. Among the 302 patients, 54.3% have heart disease.

## What the notebook does

1. Loads the data and removes the duplicate rows.
2. Explores it: class balance, correlation heatmap, age distribution by diagnosis, chest-pain type against diagnosis, and maximum heart rate by diagnosis.
3. Splits 80/20 with a fixed random seed (241 training and 61 test patients).
4. Trains a `LogisticRegression` (`max_iter=1000`) on the raw measurements.
5. Evaluates it with accuracy, a classification report, a confusion matrix and a ROC curve, and lists which inputs weigh most.

## Results

On the 61 held-out patients:

| | Precision | Recall | Patients |
|---|---|---|---|
| No disease | 0.80 | 0.83 | 29 |
| Heart disease | 0.84 | 0.81 | 32 |

**Accuracy 81.97%** and **ROC AUC 0.928**.

The inputs with the largest weights in the model are `sex` (-1.43), `thal` (-1.02), `exang` (-0.90), `ca` (-0.79) and `slope` (+0.71). The inputs were not scaled and several of them are related to each other, so read these weights as hints, not as exact importances.

## Why the duplicates matter

The same `heart.csv` is used in the [Disease-Prediction-System](https://github.com/AliHunain6234/Disease-Prediction-System) project, where Random Forest and XGBoost reach 100% test accuracy. That number is misleading: with 70.5% duplicate rows, almost every test row has an identical copy in the training data. This project avoids the problem by removing the duplicates first, which is why its score (about 82%) is lower but believable.

## Run it

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook Heart_Disease_Prediction.ipynb
```

The notebook was written in Google Colab and reads `/content/heart.csv`. In Colab, upload `heart.csv` to the session. On your own computer, change that path in the notebook to `heart.csv` (the file sits next to the notebook).

## Limitations

- **A tiny test set.** With 61 test patients, one prediction more or less moves the accuracy by about 1.6 points, so 82% is a rough figure. One split, no cross-validation.
- One simple model, with no scaling and no tuning, and no comparison with other models.
- The data has 13 measurements for a few hundred patients; it says nothing about the type or severity of a condition.
- The model is not calibrated, so its probabilities should not be read as real risk percentages.
