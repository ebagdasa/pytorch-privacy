## Differential Privacy in PyTorch

I used code from [tf-privacy](https://github.com/tensorflow/privacy) and the paper
[A General Approach to Adding Differential Privacy to Iterative Training Procedures](https://arxiv.org/abs/1812.06210).

This is a very basic implementation necessary for the quick start. There are two train commands: `train` for traditional training and `train_dp` for DP. You can see that `train_dp` has added clipping and noise to each computed gradient (gradients are not averaged over the batch at first).

To run it configure `params.yaml` and install libs: `pip install requirements.txt` and execute:

```bash
python training.py --params utils/params.yaml
``` 

The current result is 97.5% using noise multiplier 1.1 and S=1.

To count the epsilon value using RDP just use this code from the original repo:

```
!pip install tensorflow_privacy

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=60000, batch_size=250, noise_multiplier=1.3, epochs=15, delta=1e-5)
```

