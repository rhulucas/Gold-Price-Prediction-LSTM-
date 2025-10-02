**Gold Price Prediction with LSTM**
It predicts daily gold prices using a 2-layer LSTM model in PyTorch.

Overview
Data: MacroTrends gold prices (1968–2024), repo includes only sample (2023–2024).
Preprocessing: MinMax scaling, 60-day input windows.
Model: LSTM with dropout, trained with Adam + MSE.
Output: training loss curve + predicted vs. actual gold prices.

Setup: python3 -m venv myenv, source myenv/bin/activate, pip install -r requirements.txt
    
Run: python train-predict-daily-gold-prices.py, python pre_test.py                          
    
<img width="1109" height="669" alt="Figure_1" src="https://github.com/user-attachments/assets/7d23a573-bf1e-499a-8aed-2fe2c3627beb" />

<img width="767" height="413" alt="Screenshot 2025-10-02 at 2 37 39 AM" src="https://github.com/user-attachments/assets/fbd18704-48f3-44f0-8d16-26e31bc3526b" />
<img width="1022" height="171" alt="Screenshot 2025-10-02 at 2 38 04 AM" src="https://github.com/user-attachments/assets/963bc496-8aa9-4243-bfbe-cb8276d3a9f8" />






