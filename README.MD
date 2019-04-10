## Differential Privacy in PyTorch

I used code from [tf-privacy](https://github.com/tensorflow/privacy) and the paper
[A General Approach to Adding Differential Privacy to Iterative Training Procedures](https://arxiv.org/abs/1812.06210)

This is still work in progress, so the implementation might change and have bugs
that the original repo doesn't have. However, I am actively working on this code and will keep updating it.

To run it configure `params.yaml` and install libs: `pip install requirements.txt`

```bash
python training.yaml --params utils/params.yaml
``` 

To count the RDP value just use this code from the original repo: [https://github.com/tensorflow/privacy/blob/master/privacy/analysis/compute_dp_sgd_privacy.py](https://github.com/tensorflow/privacy/blob/master/privacy/analysis/compute_dp_sgd_privacy.py)

