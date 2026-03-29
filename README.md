# IPL Win Probability Predictor

A Streamlit app that estimates the live win probability of the chasing team in an IPL-style T20 match.

The project includes:

- a trained `scikit-learn` pipeline saved as `model.pkl`
- a Streamlit UI for match-state inputs and live predictions
- a training script that generates synthetic match snapshots and retrains the model

## Features

- Live win probability for the chasing side
- Confidence and momentum indicators
- Match-state insight text
- What-if simulation for the next over
- Plotly-based probability and feature charts

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- NumPy
- Plotly
- joblib

## Project Structure

```text
.
|-- app.py
|-- train_model.py
|-- model.pkl
|-- requirements.txt
`-- README.md
```

## How It Works

The app takes the current chase situation and derives a small feature set:

- `runs_left`
- `balls_left`
- `wickets_left`
- `current_run_rate`
- `required_run_rate`
- `pressure_index`
- `innings_progress`
- `batting_team`
- `bowling_team`

Those features are passed into a trained pipeline that handles preprocessing and prediction, then returns the batting side's win probability.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pinkesh2905/IPL-WinPredictor.git
cd IPL-WinPredictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Retraining the Model

If you want to regenerate the synthetic dataset and retrain the model:

```bash
python train_model.py
```

That script:

- creates synthetic IPL-style chase snapshots
- engineers numeric and categorical features
- compares multiple classifiers
- saves the best pipeline to `model.pkl`

## Model Notes

- The training data is synthetic, not scraped from real IPL ball-by-ball records.
- Predictions are illustrative and useful for demos, learning, and portfolio presentation.
- Real-world match outcomes depend on context not present in this model, such as player quality, pitch conditions, venue effects, and game pressure beyond the available features.

## Common Commands

Run the app with the local virtualenv:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Retrain with the local virtualenv:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

## License

Add a license file if you want to publish reuse terms explicitly.
