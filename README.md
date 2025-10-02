**Gold Price Prediction with LSTM**
It predicts daily gold prices using a 2-layer LSTM model in PyTorch.

Overview
Data: MacroTrends gold prices (1968–2024), repo includes only sample (2023–2024).
Preprocessing: MinMax scaling, 60-day input windows.
Model: LSTM with dropout, trained with Adam + MSE.
Output: training loss curve + predicted vs. actual gold prices.

Setup:
    python3 -m venv myenv
    source myenv/bin/activate    # Mac/Linux
    pip install -r requirements.txt

Run:
    python train-predict-daily-gold-prices.py   # full training
    python pre_test.py                          # sample demo






