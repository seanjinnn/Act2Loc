# Baselines
- TimeGeo: The method built based on the explore and preferential return (EPR) model [PNAS 2016].
Directory: baselines/timegeo.py
- Semi-Markov: In this model, the waiting time is modeled by the exponential distribution. Dirichlet prior and gamma prior are used to model the transition matrix and the intensity of the waiting time to implement a Bayesian inference [TVT 2016].
Directory: baselines/semi_markov.py
- LSTM: This model directly predicts the next locations and waiting time based on the LSTM network, and the prediction results are utilized as the synthesized trajectories [CVPR 2018].
Directory: baselines/lstm.py
- Hawkes: This model is a widely used classical temporal point process, where an occurred data point will influence the intensity function of future points [QF 2016].
Directory: baselines/hawkes.py
- MoveSim: The model proposed to synthesize human trajectories based on GAN, which introduces prior knowledge and physical regularities to the SeqGAN model [KDD 2020]
Link: https://github.com/FIBLAB/MoveSim
