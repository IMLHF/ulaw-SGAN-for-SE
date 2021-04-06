
# <img src="http://latex.codecogs.com/svg.latex?{\rm \mu}{\rm {law~SGAN~for~speech~enhancement}}" border="0" height=35/>
<img src="http://latex.codecogs.com/svg.latex?\mu{\rm {law~Spectrum~GAN}}" border="0" height=15/>

code of paper [Li, H., Xu, Y., Ke, D., & Su, K. (2021). μ-law SGAN for generating spectra with more details in speech enhancement. Neural Networks, 136, 17-27.](https://www.sciencedirect.com/science/article/abs/pii/S0893608020304421)

# Prepare for running

1) Running `bash ./nn_se/_1_perprocess.sh` to prepare data.

2) Change "root_dir" parameter in nn_se/FLAGS.py to the root of the project. For example "root_dir = /home/user/ulaw-SGAN-for-SE".

3) Ensure "PARAM = dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001" is set in last line of nn_se/FLAGS.py.

4) Running `cp nn_se dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001 -r` to create the Experiment config code dir.

# Train

Running `python -m dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001._2_train` to start training of config "dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001".

# Evaluate

Running `python -m dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001._3_enhance_testsets` to get the metrics of Experiment "dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001". The last ckpt is selected as the default ckpt to load. Alse, you can use `--ckpt` to specify the path of ckpt.

# More

See "nn_se/_1_preprocess.sh", "nn_se/_2_train.py" and "nn_se/_3_enhance_testsets.py".


