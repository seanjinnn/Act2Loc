# Methods
  ![Image](https://github.com/seanjinnn/Act2Loc/blob/main/Act2Loc.png)
# Results
  ![Image](https://github.com/seanjinnn/Act2Loc/blob/main/Trajectory%20Visualization.png)
  ![Image](https://github.com/seanjinnn/Act2Loc/blob/main/flow(Shenzhen).png)
  ![Image](https://github.com/seanjinnn/Act2Loc/blob/main/flow(Act2Loc).png)
# Baselines
- LSTM: This model directly predicts the next locations based on the LSTM network, and the prediction results are utilized as the synthesized trajectories.
Directory: baselines/LSTM.py
- SeqGan: This model is tailored for generating sequences, such as human trajectories, employing Generative Adversarial Networks (GANs). [AAAI 2017]
Directory: baselines/SeqGAN
- MoveSim: The model proposed to synthesize human trajectories based on SeqGan, which introduces prior knowledge and physical regularities to the SeqGAN model [KDD 2020]
Directory: baselines/Movesim
- W-EPR: This model integrates distance decay effects and spatial heterogeneity into the exploration phase of the EPR model, aiming to capture intra-urban dynamics.  [Physica A]
Directory: baselines/W-EPR.py
- DITRAS:  This model simulates individual trajectories by generating activity diaries with a data-driven Markov-based diary generator, and then assigning locations using an improved EPR model called d-EPR [DMKD]
Directory: baselines/DITRAS.py
# REFERENCES
- [AAAI] Yu L, Zhang W, Wang J, et al. Seqgan: Sequence generative adversarial nets with policy gradient[C]//Proceedings of the AAAI conference on artificial intelligence. 2017, 31(1).
- [KDD 2020] Feng J, Yang Z, Xu F, et al. Learning to simulate human mobility[C]//Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2020: 3426-3433.
- [DMKD] Pappalardo L, Simini F. Data-driven generation of spatio-temporal routines in human mobility[J]. Data Mining and Knowledge Discovery, 2018, 32(3): 787-829.
- [Physica A] Wang J, Dong L, Cheng X, et al. An extended exploration and preferential return model for human mobility simulation at individual and collective levels[J]. Physica A: Statistical Mechanics and Its Applications, 2019, 534: 121921.
