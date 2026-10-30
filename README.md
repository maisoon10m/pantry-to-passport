# From Pantry to Passport 🍅✈️
**Predicting cuisine from ingredient lists, and testing whether models understand food or just memorise words.**

NLP course project — Imam Abdulrahman Bin Faisal University, CCSIT (Option 4, student-proposed).

## The question
When ingredient names are replaced with verified synonyms or regional names (e.g. *eggplant → aubergine*, *zucchini → courgette*), which method keeps its Macro F1 better:
1. **TF-IDF + Logistic Regression** (traditional), or
2. **MiniLM sentence embeddings + Logistic Regression** (pretrained)?

## How to run
1. Open `Pantry_to_Passport.ipynb` in **Google Colab** (GPU runtime recommended, CPU works).
2. **Runtime → Run all.**

The notebook gets the data automatically. It checks, in order:
`data/train.json` → Kaggle public download → `MyDrive/NLP_Project/data/train.json` → an upload box.

## Data
*What's Cooking?* (Yummly / Kaggle): 39,774 recipes, 20 cuisines.
- Competition: https://www.kaggle.com/c/whats-cooking
- Public copy: https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset

The data file is **not** stored in this repository; the notebook downloads it.

## Folder structure (created when the notebook runs)
| Folder | Content |
|---|---|
| `data/` | `train.json` |
| `models/` | saved frozen models (`tfidf_lr.joblib`, `minilm_lr.joblib`) |
| `cache/` | saved MiniLM embeddings (faster second run) |
| `results/` | every table as CSV, plus `summary.json` |
| `figures/` | every figure as PNG |

## Notebook sections
Setup · Data · Cleaning · Fair split · Verified substitutions · Method 1 · Method 2 · Results · Error analysis · Team examples · **Try it** · Saved results · Sources

## Team
| Role | Name | Main technical contribution |
|---|---|---|
| Team leader | Maisoon | Data loading, cleaning and the fair train/dev/test split (Sections 1–3); coordinated the submission |
| Member 2 | Wajd | Verified substitution table, reworded test set and coverage checks (Section 4) |
| Member 3 | Jolie | Method 1: TF-IDF + Logistic Regression, dev-set tuning and preprocessing check (Section 5) |
| Member 4 | Manar | Method 2: MiniLM embeddings + Logistic Regression and the Try-it cell (Sections 6, 10) |

All members worked together on the results, the error analysis (Sections 7–8), their own examples (Section 9), and the report.

## Credits
scikit-learn, sentence-transformers (`all-MiniLM-L6-v2`, Apache-2.0), pandas, matplotlib. Parts of the code were drafted with help from Claude (Anthropic) and reviewed and tested by the team.
