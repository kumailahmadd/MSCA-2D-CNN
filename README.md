# MCSA Fault Detection using CNN

Author: 
Kumail Ahmed

---

## About

This project automatically detects and classifies faults in Squirrel Cage Induction Motors (SCIM) using Motor Stator Current Analysis (MCSA). It uses Simulink to simulate motor data, converts the signals into spectrograms, and trains a CNN to classify faults.

**See PPT for more detailed description**

The model achieves **98.1% test accuracy** across 6 fault classes.

---

## Fault Classes

- Healthy
- Broken Rotor Bar
- Bearing Fault
- Inter-turn Short Circuit
- Voltage Unbalance

---

## How It Works

1. **Data Generation** — 6 Simulink models simulate 3,000 motor signals (500 per class) with randomized frequency (49.5–50.5 Hz) and load torque (10–60 Nm).
2. **Signal Processing** — Signals are decimated to 1 kHz, DC removed, and converted to 224×224 RGB spectrograms using STFT (one channel per phase).
3. **CNN Training** — A 3-block CNN (8→16→32 filters) with Batch Normalization is trained using Adam optimizer with L2 regularization.

---

## How to Run

```matlab
% Training:
% Step 1: Generate simulation data
run('parallel_samples_generation.m')

% Step 2: Train the CNN
run('cnn_training4.m')

% Step 3: Test a single signal
run('testing_script.m')

% Or run everything at once
run('main.m')

% Model Testing:
run('testing.m') 
```

---

## Requirements

- MATLAB + Simulink
- Signal Processing Toolbox
- Deep Learning Toolbox
- Parallel Computing Toolbox

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 98.1% |
| Total Samples | 3,000 |
| Classes | 6 |

---

## Future Work

- Deploy on Raspberry Pi for real-time monitoring
- Test with real industrial motor data
