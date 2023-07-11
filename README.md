# Baselines
- LSTM: This model directly predicts the next locations based on the LSTM network, and the prediction results are utilized as the synthesized trajectories.
Directory: baselines/LSTM.py
- MoveSim: The model proposed to synthesize human trajectories based on GAN, which introduces prior knowledge and physical regularities to the SeqGAN model [KDD 2020]
Directory: baselines/code/Movesim.py

# REFERENCES
- [KDD 2020] Feng, J., et al., 2020. Learning to Simulate Human Mobility. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM. doi:10.1145/3394486.3412862.
- [DMKD] Pappalardo, L. and Simini, F. 2018. Data-driven generation of spatio-temporal routines in human mobility. Data mining and knowledge discovery, 32(3), 787-829. doi:10.1007/s10618-017-0548-4.
- [Physica A] Wang, J., et al. 2019. An extended exploration and preferential return model for human mobility simulation at individual and collective levels. Physica A: Statistical Mechanics and its Applications, 534, 121921. doi:10.1016/j.physa.2019.121921.
