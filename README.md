# ALTER

## Abstract

Understanding communication and information processing among brain regions of interest (ROIs) is highly dependent on long-range connectivity, which plays a crucial role in facilitating diverse functional neural integration across the entire brain. However, previous studies generally focused on the short-range dependencies within brain networks while neglecting the long-range dependencies, limiting an integrated understanding of brain-wide communication. To address this limitation, we propose Adaptive Long-range aware TransformER (ALTER), a brain graph transformer to capture long-range dependencies between brain ROIs utilizing biased random walk. Specifically, we present a novel long-range aware strategy to explicitly capture long-range dependencies between brain ROIs. By guiding the walker towards the next hop with higher correlation value, our strategy simulates the real-world brain-wide communication. Furthermore, by employing the transformer framework, ALERT adaptively integrates both short- and long-range dependencies between brain ROIs, enabling an integrated understanding of multi-level communication across the entire brain. Extensive experiments on ABIDE and ADNI datasets demonstrate that ALTER consistently outperforms generalized state-of-the-art graph learning methods (including SAN, Graphormer, GraphTrans, and LRGNN) and other graph learning based brain network analysis methods (including FBNETGEN, BrainNetGNN, BrainGNN, and BrainNETTF) in neurological disease diagnosis.
Cases of long-range dependencies are also presented to further illustrate the effectiveness of ALTER.

![teaser](https://anonymous.4open.science/r/ALTER-72B0/figure/figure1.jpg)

## Dependencies

- python=3.9
- cudatoolkit=11.3
- torchvision=0.13.1
- pytorch=1.12.1
- torchaudio=0.12.1
- wandb=0.13.1
- scikit-learn=1.1.1
- pandas=1.4.3
- hydra-core=1.2.0

## Installation

```html
conda create --name alter python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge wandb
```

## Usage

- Run the following command to train the model.
```html
python -m alter model=lrbgt dataset=ABIDE
