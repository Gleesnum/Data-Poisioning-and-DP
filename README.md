# Differential Privacy & Backdoor Detection in Neural Networks

This project implements and evaluates a robust anomaly detection system for identifying backdoor attacks in neural networks using **Differential Privacy (DP)**. Based on the methods in Min Du, Ruoxi Jia, and Dawn Song's ROBUST ANOMALY DETECTION AND BACKDOOR AT-
TACK DETECTION VIA DIFFERENTIAL PRIVACY (https://openreview.net/pdf?id=SJx0q1rtvS). This experiment explores how injecting noise during training can help detect poisoned data without relying on a clean validation set.

## Summary

Backdoor attacks insert small trigger patterns into training data to manipulate model predictions while preserving high test accuracy. This project replicates and extends the findings of research showing that DP-based training helps expose poisoned inputs via activation-based anomaly detection.

## Experiment Overview

- Dataset: [MNIST]
- Model: Custom CNN (`SimpleCNN`)
- Poisoning: Adds a 2x2 white square trigger in bottom-right corner of selected inputs
- Privacy Mechanism: [Opacus](https://opacus.ai/) with DP-SGD (Gaussian noise + per-sample clipping)
- Detection: Based on loss-based anomaly scores evaluated with AUROC and AUPR



