Source: https://github.com/SymbioticLab/FedScale
Revision: 7ec441c2afa99510535adebe155d89fa8bb2c637
File: examples/auxo/clustering.py
Changes: removed unused plotting and KL imports; cosine-only adapter; corrected local-to-global client index mapping after filtering known clients.
Additional change: zero-divergence batches receive finite maximal similarity rewards.
