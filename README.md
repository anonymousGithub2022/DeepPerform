<!-- # Rebuttal

## Retrain AdNN to Improve Robustness
|Subject| |Linf| ||| L2| ||
|:----|:----|:----|:----|:----|:----|:----|:----|:----|
| |Acc(before)|Acc(after)|FLOPs(before)|FLOPs(after)|Acc(before)|Acc(after)|FLOPs(before)|FLOPs(after)|
|SN_C10| 92.34| 45.67 |31.14 | 15.43 |92.34| 40.12 |31.30 | 12.34 |
|BD_C10| 91.35|  30.56| 38.39 | 17.32|91.35| 26.43| 38.39 | 17.43 |
|RN_C100| 65.43 | 30.45|181.57 | 90.23|65.43| 18.32| 182.12 |88.43|
|DS_C100| 58.78| 25.43| 157.66 | 66.42|58.78| 28.43| 157.66 | 81.43 |
|DS_SVHN| 94.54| 32.32 | 228.32 | 89.32|94.54| 17.43| 228.32 | 111.17 |

From the results, we observe that although traditional retraining method improve the efficiency robustness (FLOPs are decreased), it will hurt the AdNNs accuracy. The results indicate the standard adversarial training is not a good solution and the necessary of designing new methods.


## Performance of detecting natural perturbations.

|Subject|Gaussian Noise|Shot Noise|Impulse Noise|
|:----|:----|:----|:----|
|SK_C10|0.64 |0.65 |0.69|
|BD_C10|0.62 |0.63 |0.66|
RA_C100|0.67 |0.61 |0.68|
DS_C100|0.61 |0.60 |0.62|
DS_SVHN|0.71 |0.67 |0.77|

We evaluate the detector trained with the images from DeepPerfrom on natural perturbed images, and the results are shown in the above table.
The results show that DeepPerfrom can help identify the natural perturbed images to some degree; even the **perturbation types are unknown to the developers**. 


 -->

## Description
*DeepPerform* is designed to generate test samples that evaluate the efficiency robustness of Adaptive Neural Networks models. Specifically, *DeepPerform* perturbs the seed inputs with human unnoticable perturbations, and the perturbed inputs will consume more computational resources but keep the same semantic with the seed inputs.







## Design Overview
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DeepPerform/blob/main/fig/0001.jpg" width="800" height="300" alt="Design Overview"/><br/>
</div>    
*DeepPerform* applies the idea of GAN to generate human unnoticeable perturbations that degrade the performance of AdNNs. The design overview of *DeepPerform* is shown in the above figure. In the training phase, *DeepPerform* constructs three objective functions to learn and approximate the distribution of inputs that consume more computational resources. In addition, the GAN architecture can help *DeepPerform* generate test inputs that are close to the distribution of the original input, thus the generated test inputs are semantic equivalent to the initial seed inputs. For the detail design, please refer to our papers.

## Relation Between AdNNs' FLOPs and Performance
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DeepPerform/blob/main/fig/tmp.jpg" width="800" height="200" alt="Relationship"/><br/>
</div>    
The above figure visualizes the relationship AdNNs' FLOPs and end-to-end performance (latency and energy consumption).



## File Structure
* **src** -main source codes.
  * **./src/model_arch** -the model architecture of the tested AdNN models.
  * **./src/AbuseGan.py** -the implementation of proposd DeepPerform.
  * **./src/ilfo.py** -the implementation of ILFO.
* **trainAbuseGan.py** the script to train DeepPerform Model.
* **generate_gan_perturbation.py** -the script is used for generating GAN-based perturbations.
* **generate_ilfo_perturbation.py** -the script is used for generating ILFO perturbations.
* **test_latency.py** -this script measure the latency of the generated adversarial examples.
* **exp_gpuXX.sh** -bash script to run experiments (**XX** are integer numbers).
* **exp_gpuXX_l2.sh** -bash script to run experiments (**XX** are integer numbers).


## How to run

We provide the bash script that generate adversarial examples and measure the efficiency in **exp_gpu4.sh**. **exp_gpu5.sh**, **exp_gpu6.sh**,**exp_gpu7.sh**, **exp_gpu4_l2.sh**. **exp_gpu5_l2.sh**, **exp_gpu6_l2.sh**,**exp_gpu7_l2.sh**  are implementing the similar functionality but for different gpus. 

 So just run `bash exp_gpu4.sh` or `bash exp_gpu4_l2.sh`



## Performance Degradation Results Under different Perfubations
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DeepPerform/blob/main/fig/perturbation1.jpg" width="900" height="350" alt="Examples"/><br/>
</div> 

The above figure shows the maximum performance degradation metrics (I-FLOPs, I-Latency, I-Energy) for different experimental subjects.


## Ovheads of *DeepPerform*
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DeepPerform/blob/main/fig/overhead.jpg" width="900" height="300" alt="Examples"/><br/>
</div> 
The above figure shows average overheads of the generating one test sample for *DeepPerform* and *ILFO*. From the results, we observe that *DeepPerform* costs 6-8 milliseconds to generate one test sample while *ILFO* cost more than 60 seconds.


## Semantic of the Generated Test Samples
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DeepPerform/blob/main/fig/00012.jpg" width="900" height="300" alt="Examples"/><br/>
</div>   
