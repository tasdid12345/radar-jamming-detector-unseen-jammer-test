# Class-balanced LightGBM vs MLP for jamming detection in simulated radar returns

Radar jamming detection from target returns is a key sensing task for joint radar-communication systems, as motivated by "Deep learning assisted radar jamming detection from target returns for joint radar communication systems" (2024). That paper proposes a deep-learning detector; this repository does not reproduce it, and instead asks whether a simpler, feature-based gradient-boosted model handles jammer types it never saw in training.

**Question:** Can a class-balanced LightGBM detector on spectral and statistical features of simulated radar returns beat a standard MLP deep-learning-style detector at flagging jamming, especially when the jammer type is unseen during training?

**Dataset:** Simulated complex baseband radar pulses (256 samples each) generated with numpy at run time: a target echo at random range and SNR plus noise, with barrage and spoofing-style jamming used for training and a held-out chirp jammer plus a low jamming-to-target slice in the test set. This is a simple simulation, not the paper's own dataset (`same_as_paper: false`).

**Baseline:** a scikit-learn MLPClassifier on raw normalised amplitude and phase samples, standing in for the paper's deep-learning detector.

**Improved model:** a class-balanced LightGBM on handcrafted spectral and statistical features, with CV-tuned hyper-parameters and isotonic calibration. Both models get the same CV tuning and calibration.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (figure in `figures/`).

## Reference

This project is motivated by the following paper:

> Z Xu, I Petrunin, A Tsourdos (2024). *Deep learning assisted radar jamming detection from target returns for joint radar communication systems*. 2024 International Radar. https://ieeexplore.ieee.org/abstract/document/10994104/

Paper author: Prof. Antonios Tsourdos, Cranfield University (a.tsourdos@cranfield.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
