# Ablations for NORD-F

### Basic Contributions

1. I have made a minimal implementation which is easy to use.
2. Saliency Maps for classification and gradient reversal branch
3. Test better loss functions for Reconstruction
4. Change backbone to Latest ConvNet --> ConvNeXtv2

### Short analysis

1. L1 loss is much for stable than MSE. 
- Maybe we can try Huber Loss next.
2. ConvNeXtv2 backbone is much more stable than VGG [In my opinion, better than ViT]

### Current Issues:

1. L1 is better than MSE for a perceptual loss -- [Potential Reason](https://towardsdatascience.com/perceptual-losses-for-image-restoration-dd3c9de4113#:~:text=L1%20has%20constant%20gradients%2C%20which,with%20L2%20and%20L1%20losses.)
2. I think the model is struggling with smaller batch sizes because:

```        # Checking if values are within the range [0, 1]
        if torch.any(f < 0) or torch.any(f > 1):
            print("Rejecting sample due to out-of-range values:", f)
            return None, None, None
```
Gives:
```
Rejecting sample due to out-of-range values: tensor([ 0.0000e+00, -1.1921e-07], device='cuda:0')
Traceback (most recent call last):
  File "/home/localssk23/Downloads/ishika/basic_train.py", line 306, in <module>
    results[key] = train_model(backbone, upper_branch, classification_branch, reconstruction_branch, train_loader, val_loader, criterion_sim, criterion_cls, criterion_recon, optimizer, lambda_sim, lambda_cls, lambda_recon)
  File "/home/localssk23/Downloads/ishika/basic_train.py", line 229, in train_model
    sim_loss = criterion_sim(similarity_score, labels.float())
  File "/home/localssk23/backup/soumya/env/ppml/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/localssk23/backup/soumya/env/ppml/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/localssk23/backup/soumya/env/ppml/lib/python3.10/site-packages/torch/nn/modules/loss.py", line 621, in forward
    return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)
  File "/home/localssk23/backup/soumya/env/ppml/lib/python3.10/site-packages/torch/nn/functional.py", line 3162, in binary_cross_entropy
    if target.size() != input.size():
AttributeError: 'NoneType' object has no attribute 'size'
```

### Remaining Things:

1. Which baselines should I set up for ood? 
2. What metrics do I need to report? -- Should I just follow the one in the paper?
3. A deep dive into analysis the Saliency maps

For all the code (and how to use it) -- please check the github.

### Notes
1. For basic run, `python basic_train.py`
2. For plotting everything `python train.py`
3. Ignore misc folder