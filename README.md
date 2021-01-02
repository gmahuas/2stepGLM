# 2stepGLM - Python example using TensorFlow

Code illustrating the two-step inference approach for the Generalized Linear Models detailed in the following paper:
G. Mahuas, G. Isacchini, O. Marre, U. Ferrari and T. Mora. A new inference approach for training shallow and deep generalized linear models of noisy interacting neurons. Accepted for spotlight presentation at NeurIPS2020.

The example.ipynb jupyter notebook contains the code to simulate and compare the models that have been inferred using the 2step_inference.ipynb (inference of the interaction model of the GLM using the two-step approach) and model_inference.ipynb (inference of the LN model and whole maximum likelihood inference of the GLM) notebooks.
These scripts use several common libraries including: tensorflow, numpy, matplotlib and pickle5, be sure to have them installed prior to running the notebooks.
