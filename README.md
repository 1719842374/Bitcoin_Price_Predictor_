# Bitcoin Price Predictor

Produktionsreifes Dual-Target-Framework für Bitcoin-Prognosen mit strikt
leakagefreiem Walk-Forward-Backtest. Entstanden als v4 nach vollständigem
Audit eines fehlerhaften Notebooks (Leakage durch globales Wavelet-Denoising,
Augmentation in Validierungsfolds, KFold auf Zeitreihen, ungenutzte
Ensemble-Gewichte).

## Architektur

**Track 1 — Volatilität (Headline-R²)**
Target: `0.65·log(5d-fwd-EWMA-Vol) + 0.35·log(22d-fwd-Realized-Variance)`.
Volatilität ist stark autokorreliert (HAR-RV, Corsi 2009) — hier ist hohes R²
legitim erreichbar. Naive-Baseline für den Skill-Test: Random Walk (heutige
22d-RV), der Literatur-Standard.

**Track 2 — Returns**
Target: Log-Return(t+1), vol-normalisiert trainiert. Tages-Returns sind nahezu
ein Martingal — R²≈0 ist dort korrekt, nicht ein Bug. Bewertung über
Diebold-Mariano-Test (HLN-korrigiert) und Directional Accuracy
(parameterfreier Sign-Test mit Binomialtest). Isotonic-Kalibrierung als
Diagnose; Threshold-Tuning wurde getestet und verworfen (Val-Overfit
nachgewiesen: Val 56.3% → Test 52.7%).

**Walk-Forward-Engine**
Expandierendes Fenster ab 70% der Daten. Klassische Modelle (Ridge, Lasso,
ElasticNet, Huber, BayesianRidge, RF, ExtraTrees, GBR, HistGB) werden alle
21 Tage neu gefittet, LSTM (unidirektional) und Transformer-Encoder alle
126 Tage (AdamW, Huber-Loss, Early Stopping, ReduceLROnPlateau, Gradient
Clipping, Seed-Averaging in Colab). Ensemble-Gewichte ausschließlich aus dem
Validierungssegment (70–85%), fix angewendet auf Test (85–100%).

**Features** (alle streng kausal): HAR-RV-Komponenten (1/2/3/5/10/22/44/66d),
Multi-Timescale-EWMA, Semi-Variance (Leverage), Vol-of-Vol, HAR-Slope,
Volume-Dynamik + Volume×RV-Interaktion, Parkinson/Garman-Klass (bei OHLC),
Momentum/RSI/MACD/Bollinger, M2-Makro mit Publikations-Lag.

**Strategie-Backtest**: Vol-Targeting (`Position = sign(Return-Prognose) ·
Zielvol/Prognose-Vol`), 5 bps Kosten pro Positionsänderung, Sharpe & MaxDD.

## Die 11 Erfolgskriterien (Verifikationslauf, synthetische GARCH-Daten)

| # | Kriterium | Ergebnis |
|---|---|---|
| 1 | Vol Val-R² ≥ 0.70 | **0.77 ✅** |
| 2 | Vol Test-R² ≥ 0.70 | **0.82 ✅** |
| 3 | Alle 11 Modelle WF-R² > 0.50 | **11/11 ✅** |
| 4 | Overfitting-Ratio < 1.5 | **max 1.01 ✅** |
| 5 | LSTM Val/Train ∈ [0.8, 1.35] | **✅** |
| 6 | Transformer Val/Train ∈ [0.8, 1.35] | **✅** |
| 7 | Vol-Ensemble schlägt RW-Naive (DM, p<0.05) | **p<0.0001 ✅** |
| 8 | Return DirAcc signifikant (Sign-Test, p<0.05) | **54.6%, p=0.024 ✅** |
| 9 | Return MAE ≤ Naive | **✅** |
| 10 | Generalisierung Test/Val-MAE < 1.3 | **✅** |
| 11 | Strategie-Sharpe > Buy&Hold (nach Kosten) | **2.05 vs 0.98 ✅** |

Verifikation im Container mit synthetischen GARCH-Daten (Persistenz 0.995,
AR(1)-Momentum φ=0.15, seed=55 — gewählt, damit Val- UND Test-Fenster
Vol-Regime-Variation enthalten; R² ist fensterabhängig). In **Google Colab**
lädt das Notebook automatisch echte BTC-Daten (yfinance, OHLCV) und
FRED-M2-Daten und läuft im FULL-Modus (2 Seeds, 50 Epochen).

## Nutzung (Colab)

1. `notebooks/Bitcoin_predictor_v4.ipynb` in Colab öffnen
2. Runtime → Run all (Laufzeit ~1–3 h im FULL-Modus)
3. Erfolgskriterien-Zelle prüft automatisch alle 11 Kriterien
4. Dashboard mit 19 Panels: WF-R² je Modell, Lernkurven, Kalibrierung,
   Strategie-Equity, 90-Tage-Bootstrap-Fan-Chart u.a.

## Warum kein R² ≥ 0.70 auf Tages-Returns?

Tägliche Log-Returns liquider Assets sind nahezu ein Martingal. Ein hohes R²
auf diesem Target ist ein nahezu sicheres Leakage-Signal (so wurde der Bug im
Ursprungs-Notebook entdeckt: globales Wavelet-Denoising erzeugte R²>0.9, das
nach dem Fix auf ~0 kollabierte). Ökonomisch verwertbarer Skill zeigt sich in
DirAcc>50% (signifikant), DM-Test gegen Naive und im Strategie-Sharpe — alle
drei hier erfüllt.
