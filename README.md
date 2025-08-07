# Episodic and associative memory from spatial scaffolds in the hippocampus

This is the official implementation of [Episodic and associative memory from spatial scaffolds in the hippocampus](https://www.nature.com/articles/s41586-024-08392-y). In this code we implement Vector-HaSH (Vector Hippocampal Scaffolded Heteroassociative Memory), a neocortical-entorhinal-hippocampal network that implements a high-capacity general associative memory, spatial memory and episodic memory.

## 📚 Documentation

- **[详细中文文档 / Detailed Chinese Documentation](README_DETAILED.md)** - 完整的中文说明文档
- **[Detailed English Documentation](README_DETAILED_EN.md)** - Complete English documentation

## 🚀 Quick Start

### Recommended Running Order
1. `Scaffold_testing_Fig_2.ipynb` - Understand basic scaffold mechanisms
2. `autoencoder_assoc_mem_Fig_3.ipynb` - Learn associative memory principles  
3. `11Rooms_11Maps_Fig_4.ipynb` - Explore spatial mapping capabilities
4. `Full_model_testing_Fig_3.ipynb` - Complete model testing
5. `sequence_autoencoder_Fig_5.ipynb` - Sequence memory experiments

### Key Files
- **Main Experiments**: `*_Fig_*.ipynb` notebooks correspond to paper figures
- **Core Algorithms**: `src/assoc_utils_np_2D.py`, `src/seq_utils.py`
- **Analysis Tools**: `MTT.py`, `Plots_for_baseline_*.py`

## Citation
If you use this code for your research, please cite our paper.
```
﻿@Article{Chandra&Sharma2025,
author={Chandra, Sarthak
and Sharma, Sugandha
and Chaudhuri, Rishidev
and Fiete, Ila},
title={Episodic and associative memory from spatial scaffolds in the hippocampus},
journal={Nature},
year={2025},
month={Jan},
day={15},
issn={1476-4687},
doi={10.1038/s41586-024-08392-y},
url={https://doi.org/10.1038/s41586-024-08392-y}
}

```
