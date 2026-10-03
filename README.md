# Baseline Predictive Pipeline -- ETAI

> 20260675 Carolina Raposo

## Week 2:

| | Train acc. | Test acc. | Gap (train - test) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.679 | 0.680 | -0.001 | 0.66 | 0.60 | 0.63 |
| **Decision Tree** | 0.829 | 0.629 | +0.199 | 0.62 | 0.49 | 0.55 |

### Model Comparison:
The Decision Tree model overfitted because the train accuracy (0.829) is much higher than the test accuracy (0.629), 
showing a +0.199 gap, while the Logistic Regression model maintained a near-zero gap (-0.001). 
The Decision Tree classification report shows lower precision, recall, and F1-score compared to Logistic 
Regression. Specifically, the drop in recall from 0.60 to 0.49 indicates that the Decision Tree missed more than half of
the individuals who actually recidivated (false negatives). Overall, Logistic Regression proved to be more reliable, 
offering both better generalization and superior detection of positive cases.

Fairness (False Positive Rate by Race):
- African-American (n=303): COMPAS = 0.44 | Logistic Regression = 0.33 | Decision Tree = 0.27
- Caucasian (n=232): COMPAS = 0.25 | Logistic Regression = 0.23 | Decision Tree = 0.24

COMPAS has a big gap between Black and White defendants (0.44 vs 0.25). Logistic regression reduces this gap to 0.33 vs 
0.23. The Decision Tree lowers the gap even more, but mostly because its recall dropped and it predicts fewer positive 
cases overall.


## Week 3 (after the cleaning pipeline):

| | Train acc. | Test acc. | Gap (train - test) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.678 | 0.655 | +0.023 | 0.65 | 0.51 | 0.57 |
| **Decision Tree** | 0.799 | 0.604 | +0.195 | 0.59 | 0.43 | 0.50 |

### Before vs. after cleaning:
Both models lost a little test accuracy (LR 0.680 -> 0.655, DT 0.629 -> 0.604) and recall (LR 0.60 -> 0.51,
DT 0.49 -> 0.43). The comparison is not like-for-like: cleaning removed duplicates and turned invalid values and
placeholders into missing values, and the pipeline still drops every row with a missing value, so the test set
shrank from 1252 to 1188 rows. Cleaning alone does not improve the metrics; imputation should recover the lost rows.
It did fix the `race` column (16 fragmented groups -> 6), so the fairness numbers are now reliable.

### Model Comparison:
Logistic Regression is still the better model (higher accuracy, recall and F1) and barely overfits (gap +0.023),
while the Decision Tree still overfits heavily (gap +0.195), since it has no depth limit.

Fairness (False Positive Rate by Race, cleaned groups):
- African-American (n=299): COMPAS = 0.47 | Logistic Regression = 0.30 | Decision Tree = 0.33
- Caucasian (n=241): COMPAS = 0.23 | Logistic Regression = 0.16 | Decision Tree = 0.19

The gap is +0.24 for COMPAS and +0.14 for both models: reduced, not eliminated. This corrects the week 2
conclusion that the Decision Tree nearly closed the gap, which was largely noise from the fragmented race groups.

### Week 4: preprocessing inside the pipeline + cross-validation

The pipeline now cleans the data, sets a locked test set aside (20%, never used yet), and evaluates one `Pipeline`
(imputation + target encoding + robust scaling + model) on the development set with both a single holdout split
and stratified 5-fold cross-validation. Everything fitted (medians, encoder means, scaling) is learned on training
rows only, inside each fold.

| Model | Holdout acc. (W3) | Holdout acc. (W4) | CV acc. (mean ± std) | CV train - val gap |
|---|---|---|---|---|
| Dummy (majority) | - | 0.550 | 0.549 ± 0.000 | -0.000 |
| Logistic Regression | 0.655 | 0.674 | 0.672 ± 0.013 | +0.003 |
| Decision Tree | 0.604 | 0.591 | 0.610 ± 0.018 | +0.085 |
| Random Forest | - | 0.644 | 0.650 ± 0.018 | +0.083 |

W3 = naive preprocessing evaluated on the test split; W4 = new preprocessing on a validation split of the development set,
so the two holdout columns are indicative, not exactly comparable.

**Preprocessing vs. week 3:** Logistic Regression improved (0.655 -> 0.674); the Decision Tree moved within noise
(0.604 -> 0.591, CV std 0.018), though its overfitting gap dropped from +0.195 to +0.085.

**Holdout vs. CV:** a single holdout number depends on the split: the 5 folds of the Logistic Regression range from 0.654 to 0.688.
I trust the CV mean ± std more, since it uses every development row for validation and shows how much the estimate moves.
Both agree on the ranking.

**Best model:** Logistic Regression still holds under CV: it beats the Random Forest (0.672 vs 0.650) in all 5 folds
and barely overfits, so the week 2/3 conclusion stands. Every real model beats the 0.549 majority-class floor.

**Fairness (FPR by race, out-of-fold):** the African-American vs. Caucasian gap is +0.22 for COMPAS and +0.13 / +0.09 / +0.13
for Logistic Regression / Decision Tree / Random Forest: reduced, not eliminated. (The dummy's FPR of 0 is meaningless:
it never predicts recidivism.)

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── data_diagnostics.py # checks for invalid values
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py            # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```



## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
