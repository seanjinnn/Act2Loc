# Baselines
- LSTM: This model directly predicts the next locations and waiting time based on the LSTM network, and the prediction results are utilized as the synthesized trajectories [CVPR 2018].
Directory: baselines/LSTM.py
- MoveSim: The model proposed to synthesize human trajectories based on GAN, which introduces prior knowledge and physical regularities to the SeqGAN model [KDD 2020]
Directory: baselines/code/Movesim.py

# REFERENCES
[KDD 2020] Feng, J., et al., 2020. Learning to Simulate Human Mobility. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM. doi:10.1145/3394486.3412862.
