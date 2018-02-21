'''
The all-important kernels and their hyperparameters.

Created on Dec 11, 2014

@author: ntraft
'''
from __future__ import division
import numpy as np

from gp import ParametricGPModel
import covariance as cov
#from com.ntraft.gp import ParametricGPModel
#import com.ntraft.covariance as cov


# Hyperparameters from seq_eth #175
model1 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.5128), np.exp(2*5.3844)),
	cov.linear_kernel(np.exp(-2*-2.8770)),
	cov.noise_kernel(np.exp(2*-0.3170))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.2839), np.exp(2*2.5229)),
	cov.linear_kernel(np.exp(-2*-4.8792)),
	cov.noise_kernel(np.exp(2*-0.2407))
))

# Hyperparameters from seq_eth #48 - HUGE variance and smooth
model2 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.0194), np.exp(2*2.7259)),
	cov.linear_kernel(np.exp(-2*-3.2502)),
	cov.noise_kernel(np.exp(2*-1.1128))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.5181), np.exp(2*5.4197)),
	cov.linear_kernel(np.exp(-2*-0.8087)),
	cov.noise_kernel(np.exp(2*-0.5089))
))

# Hyperparameters from seq_eth #201 - pretty squirrely
model3 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.8777), np.exp(2*6.2545)),
	cov.linear_kernel(np.exp(-2*-1.6083)),
	cov.noise_kernel(np.exp(2*0.1863))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.0143), np.exp(2*3.4259)),
	cov.linear_kernel(np.exp(-2*-5.5210)),
	cov.noise_kernel(np.exp(2*-0.2941))
))

# Hyperparameters from seq_eth #194 - BAD
model4 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(1.1525), np.exp(2*1.8115)),
	cov.linear_kernel(np.exp(-2*-4.5797)),
	cov.noise_kernel(np.exp(2*-6.1552))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(1.1738), np.exp(2*1.7332)),
	cov.linear_kernel(np.exp(-2*-5.3511)),
	cov.noise_kernel(np.exp(2*-6.2679))
))

# Hyperparameters for seq_hotel.
model5 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.0257), np.exp(2*2.8614)),
	cov.linear_kernel(np.exp(-2*-5.5200)),
	cov.noise_kernel(np.exp(2*0.5135))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.0840), np.exp(2*2.3497)),
	cov.linear_kernel(np.exp(-2*-6.1052)),
	cov.noise_kernel(np.exp(2*-0.1758))
))

# Trained from [48, 175].
model6 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.8541), np.exp(2*6.0068)),
	cov.linear_kernel(np.exp(-2*-1.5976)),
	cov.noise_kernel(np.exp(2*-0.3789))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.4963), np.exp(2*5.1759)),
	cov.linear_kernel(np.exp(-2*-0.6863)),
	cov.noise_kernel(np.exp(2*-0.3095))
))

# Trained from 319:331 jointly.
model7 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.8782), np.exp(2*5.8351)),
	cov.linear_kernel(np.exp(-2*1.4428)),
	cov.noise_kernel(np.exp(2*0.3991))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.7552), np.exp(2*4.0612)),
	cov.linear_kernel(np.exp(-2*-7.0977)),
	cov.noise_kernel(np.exp(2*0.1184))
))

# Trained from 319:331 greedily.
model8 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.5479), np.exp(2*5.6035)),
	cov.linear_kernel(np.exp(-2*4.8811)),
	cov.noise_kernel(np.exp(2*0.2929))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.2256), np.exp(2*5.5083)),
	cov.linear_kernel(np.exp(-2*4.3907)),
	cov.noise_kernel(np.exp(2*0.0081))
))

# Trained from 169:174 jointly.
model9 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(4.0182), np.exp(2*6.2240)),
	cov.linear_kernel(np.exp(-2*0.0586)),
	cov.noise_kernel(np.exp(2*-0.1050))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.4996), np.exp(2*5.1338)),
	cov.linear_kernel(np.exp(-2*0.0710)),
	cov.noise_kernel(np.exp(2*-0.1966))
))

# Trained from [48:49, 169:170, 172:175, 319:331] jointly. - Good, perhaps a bit noisy.
model10 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.9542), np.exp(2*5.9956)),
	cov.linear_kernel(np.exp(-2*1.8273)),
	cov.noise_kernel(np.exp(2*0.2926))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.8671), np.exp(2*4.1996)),
	cov.linear_kernel(np.exp(-2*-5.7726)),
	cov.noise_kernel(np.exp(2*0.0360))
))

# Trained from [48:49, 169:170, 172:175, 319:331] greedily. - BAD
# Can be made s.p.d. with a large addition if you need to show a bad example.
model11 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.5479), np.exp(2*5.6035)),
	cov.linear_kernel(np.exp(-2*5.3545)),
	cov.noise_kernel(np.exp(2*0.2929))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(13.1823), np.exp(2*5.7036)),
	cov.linear_kernel(np.exp(-2*-0.9699)),
	cov.noise_kernel(np.exp(2*2.0438))
))

# Trained from [48:49, 169:170, 172:175, 319:331], averaged. - Nice.
model12 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.4171), np.exp(2*4.7080)),
	cov.linear_kernel(np.exp(-2*-2.7809)),
	cov.noise_kernel(np.exp(2*0.0088))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.4912), np.exp(2*3.6663)),
	cov.linear_kernel(np.exp(-2*-2.2506)),
	cov.noise_kernel(np.exp(2*-0.4789))
))

# Trained from [48:49, 169:170, 172:175, 319:331], average of absolute values.
model13 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.4171), np.exp(2*4.7080)),
	cov.linear_kernel(np.exp(-2*3.0120)),
	cov.noise_kernel(np.exp(2*0.3684))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.6939), np.exp(2*3.6930)),
	cov.linear_kernel(np.exp(-2*4.4675)),
	cov.noise_kernel(np.exp(2*0.5707))
))

# Trained from [194,197] jointly. Not enough. See y's linear param.
model14 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.3319), np.exp(2*5.8461)),
	cov.linear_kernel(np.exp(-2*-0.4652)),
	cov.noise_kernel(np.exp(2*0.1134))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.4688), np.exp(2*3.8112)),
	cov.linear_kernel(np.exp(-2*-5.4513)),
	cov.noise_kernel(np.exp(2*-0.2972))
))

# Trained from [194:197] jointly. Still not enough.
model15 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.4635), np.exp(2*5.7321)),
	cov.linear_kernel(np.exp(-2*0.1474)),
	cov.noise_kernel(np.exp(2*0.0922))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.8061), np.exp(2*4.1907)),
	cov.linear_kernel(np.exp(-2*-5.3629)),
	cov.noise_kernel(np.exp(2*-0.1183))
))

# Trained from [194:202] jointly.
model16 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.6491), np.exp(2*5.8470)),
	cov.linear_kernel(np.exp(-2*0.9388)),
	cov.noise_kernel(np.exp(2*0.1761))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.1300), np.exp(2*4.9362)),
	cov.linear_kernel(np.exp(-2*0.9296)),
	cov.noise_kernel(np.exp(2*-0.1385))
))

# Trained from [194:202], averaged.
model17 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.2257), np.exp(2*4.3906)),
	cov.linear_kernel(np.exp(-2*-3.5695)),
	cov.noise_kernel(np.exp(2*-0.9939))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.5265), np.exp(2*3.2624)),
	cov.linear_kernel(np.exp(-2*-4.0408)),
	cov.noise_kernel(np.exp(2*-0.9181))
))

# Trained from [194:202], in greedy cycles.
model18 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(4.4284), np.exp(2*6.5746)),
	cov.linear_kernel(np.exp(-2*3.2876)),
	cov.noise_kernel(np.exp(2*-0.5307))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(0.5139), np.exp(2*0.8271)),
	cov.linear_kernel(np.exp(-2*-5.4365)),
	cov.noise_kernel(np.exp(2*-6.2657))
))

# Trained from [194:202], in 250 rounds of early stopping (5) cycles.
model19 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.4168), np.exp(2*4.3343)),
	cov.linear_kernel(np.exp(-2*-5.8574)),
	cov.noise_kernel(np.exp(2*-0.5645))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.6582), np.exp(2*2.2945)),
	cov.linear_kernel(np.exp(-2*-5.4133)),
	cov.noise_kernel(np.exp(2*-0.3892))
))

# Trained from [194:202], in 250 rounds of early stopping (10) cycles.
model20 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(4.7982), np.exp(2*6.2814)),
	cov.linear_kernel(np.exp(-2*-2.3599)),
	cov.noise_kernel(np.exp(2*-0.2492))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.2023), np.exp(2*2.6502)),
	cov.linear_kernel(np.exp(-2*-5.5290)),
	cov.noise_kernel(np.exp(2*-0.2721))
))

# Trained from [194:224], averaged. - BEST
model21 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.3434), np.exp(2*4.5640)),
	cov.linear_kernel(np.exp(-2*-2.9756)),
	cov.noise_kernel(np.exp(2*-0.2781))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.4624), np.exp(2*3.1776)),
	cov.linear_kernel(np.exp(-2*-3.4571)),
	cov.noise_kernel(np.exp(2*-0.3478))
))

# Trained from [174:182, 184:191, 194:224], averaged. - bad, I think
model22 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.9880), np.exp(2*4.0357)),
	cov.linear_kernel(np.exp(-2*-3.3105)),
	cov.noise_kernel(np.exp(2*-0.3648))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.3049), np.exp(2*2.9303)),
	cov.linear_kernel(np.exp(-2*-3.6089)),
	cov.noise_kernel(np.exp(2*-0.3195))
))

# Hyperparameters from seq_eth #195
model23 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(5.2662), np.exp(2*7.2229)),
	cov.linear_kernel(np.exp(-2*-2.4491)),
	cov.noise_kernel(np.exp(2*0.2786))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.2134), np.exp(2*2.9315)),
	cov.linear_kernel(np.exp(-2*-5.4611)),
	cov.noise_kernel(np.exp(2*-0.3647))
))

# Hyperparameters from seq_eth #198
model24 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.3244), np.exp(2*2.8775)),
	cov.linear_kernel(np.exp(-2*-5.7386)),
	cov.noise_kernel(np.exp(2*-2.2338))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(3.6438), np.exp(2*5.8213)),
	cov.linear_kernel(np.exp(-2*1.2371)),
	cov.noise_kernel(np.exp(2*-0.3983))
))

# Hyperparameters from seq_eth #199
model25 = ParametricGPModel(
cov.summed_kernel(
	cov.matern_kernel(np.exp(4.2322), np.exp(2*5.6138)),
	cov.linear_kernel(np.exp(-2*-2.8974)),
	cov.noise_kernel(np.exp(2*-0.6104))
),
cov.summed_kernel(
	cov.matern_kernel(np.exp(2.1396), np.exp(2*1.9783)),
	cov.linear_kernel(np.exp(-2*-5.3199)),
	cov.noise_kernel(np.exp(2*-0.5393))
))

