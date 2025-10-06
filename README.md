# FMCNS
This is our Pytorch implementation for "FMCNS: Flow Matching with Causal-aware Negative Sampling for Multimodal Recommendation":  

## Overview of FMCNS
We propose a novel Flow Matching with Causal-aware Negative Sampling framework (FMCNS), which addresses the fundamental challenges of negative sampling and generative modeling in multimodal recommendation. Our framework comprises two key modules working synergistically to enhance recommendation quality.

1) Conditional Flow Matching Enhanced Generation constructs semantic graphs from heterogeneous visual and textual features, then employs deterministic straight-line flow trajectories to generate high-quality enhanced embeddings in the latent space. By leveraging frequency-based priors to reflect collaborative patterns, this module learns stable flow dynamics that enable efficient sampling with significantly fewer steps while incorporating rich semantic information from multimodal modalities.

2) Causal-aware Negative Sampling introduces a principled causal intervention approach that explicitly breaks spurious correlations between item popularity and selection probability through frequency-based debiasing. By integrating flow-enhanced item representations with causal intervention weights, this module produces semantically meaningful hard negatives while mitigating popularity bias, ensuring the model learns invariant user preferences rather than environment-dependent correlations.

Extensive experiments conducted on three publicly available datasets (Baby, Sports, and Clothing) demonstrate that our proposed FMCNS framework outperforms several state-of-the-art multimodal recommendation baselines across multiple evaluation metrics.

## Environment Requirement

* python == 3.12.6
* pytorch == 2.4.1
* numpy == 1.26.0

We provide three processed datasets: Baby, Sports, and Clothing.

## Dataset Statistics
| Dataset   | #Users | #Items | #Interactions | Sparsity | Modality |
|-----------|--------|--------|---------------|----------|----------|
| Baby      | 19,445 | 7,050  | 160,792       | 99.88%   | V, T     |
| Sports    | 35,598 | 18,357 | 296,337       | 99.95%   | V, T     |
| Clothing  | 39,387 | 23,033 | 278,677       | 99.97%   | V, T     |

## Example to Run the Codes

The instructions for the commands are clearly stated in the codes.

* Baby dataset
```
python main.py -m FMCNS -d baby
```

* Sports dataset

```
python main.py -m FMCNS -d sports
```

* Clothing dataset

```
python main.py -m FMCNS -d clothing
```


## FMCNS
The released code consists of the following files.
```
--images
--data
    --baby
    --clothing
    --sports
--dataHandler.py           
--main.py
--model.py
--params.py
--logger.py
--utils.py
```

## Acknowledgement
The structure of this code is inspired by the [MMRec](https://github.com/enoche/MMRec) framework. We acknowledge and appreciate their valuable contributions.