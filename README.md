```markdown
# MINIC3 Dual‑Task Prediction Tool

This repository contains the complete code and simulated data for the paper:

**“A Python‑Based Interactive Web Tool for Dual‑Task Prediction of Treatment Response and Adverse Events in MINIC3 Immunotherapy: A Proof‑of‑Concept Study”**  
*(under review)*

- **Live web app**: [https://minic3-predictor-f3fxplj5xpfbzwddntw2bu.streamlit.app/](https://minic3-predictor-f3fxplj5xpfbzwddntw2bu.streamlit.app/)  
- **Corresponding author**: Puyao Sun ([sunpuyao@stu.zzu.edu.cn](mailto:sunpuyao@stu.zzu.edu.cn))

---

## 📁 Repository structure

```
.
├── main.py                         # Streamlit web application
├── generate_simulated_data.py      # Script to reproduce the simulated dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT license
```

---

## 🔧 Reproduce the simulated dataset

All analyses in the paper are based on a **fully reproducible simulated dataset**.

To generate the exact dataset used in the study:

```bash
python generate_simulated_data.py
```

This will create `simulated_data.csv` with 2 000 patients, including:

- demographics (age, gender)
- clinical features (ECOG, metastases, prior therapies)
- biomarkers (PD‑L1, TMB, NLR, LDH, CRP, albumin)
- simulated outcome labels (response, adverse events, PFS)

The generation logic is explicitly described in the **Methods** section of the paper (Section 2.2).

---

## 🚀 Run the web application locally

1. Clone this repository:  
   ```bash
   git clone https://github.com/spy929/minic3-predictor.git
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:  
   ```bash
   streamlit run main.py
   ```

You will then be able to use the same dual‑task prediction interface locally.

---

## 📊 What the tool does

- Predicts **treatment response probability** and **adverse event risk** for MINIC3 immunotherapy
- Uses two independent random forest models
- Provides an interactive interface (no coding required)
- Fully open‑source and ready for adaptation to other immunotherapies

---

## ⚠️ Important note

This tool is a **proof‑of‑concept framework** for dual‑task prediction.  
It is **not intended for clinical decision‑making** and has not been validated on real‑world patient data.

---

## 📄 License

MIT — you are free to use, modify, and distribute this code with attribution.

---

## ✒️ Citation

If you use this code or the simulated dataset in your work, please cite the original paper (citation to be added upon publication).

---

## 🙏 Acknowledgments

This work was supported by the Henan Province Zhongyuan Medical Science and Technology Innovation Development Fund (Grant No. ZYYC2503201‑8).
```
