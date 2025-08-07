# Episodic and associative memory from spatial scaffolds in the hippocampus

This is an **annotated implementation** of the official code for [Episodic and associative memory from spatial scaffolds in the hippocampus](https://www.nature.com/articles/s41586-024-08392-y Episodic and associative memory from spatial scaffolds in the hippocampus Episodic and associative memory from spatial scaffolds in the hippocampus).
In this enhanced version, we provide detailed documentation for Vector-HaSH (Vector Hippocampal Scaffolded Heteroassociative Memory) – a neocortical-entorhinal-hippocampal network model for high-capacity associative, spatial, and episodic memory.

Key Additions:
🧠 **Comprehensive code annotations** explaining critical logic and mathematical operations
📊 **Dataflow diagrams** visualizing architecture and tensor operations
🔍 **Parameter specifications** with shape definitions and semantic analysis
📖 **Markdown guides** bridging theoretical concepts to implementation details


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
If you use this code for your research, please cite the orginal paper.
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
