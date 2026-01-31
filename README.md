# WAVIER

This repository contains all relevant codes and materials prepared for our work, _"WAVIER: Wavelet-based Autoencoding Variational Autoencoder for Regression for Pupillometry-based Cognitive Workload Estimation"_.

WAVIER is a wavelet-assisted method for adapting variational autoencoders (VAEs) to length-invariant reconstruction tasks with downstream regression objectives. WAVIER's frequency-domain VAE is retrofitted with a conditional prior for regression, a probabilistic regression head and a auxiliary-variable regularizer, allowing it to output a latent space with clear separations denoted by a regression objective and ensuring self-consistency of the architecture.

This model was evaluated with the [COLET Dataset](https://zenodo.org/records/5913227), an open dataset collecting live gaze tracking and pupillometry data from 47 participants while participating in a Captcha Game. The NASA Task Load Index questionnaire was administered after each experimental section, which we used as a baseline for cognitive workload estimation via regression.
