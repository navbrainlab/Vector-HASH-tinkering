# Vector-HaSH: Detailed Documentation (English)

## Project Overview

This is the official implementation of the Nature paper "Episodic and associative memory from spatial scaffolds in the hippocampus". The project implements Vector-HaSH (Vector Hippocampal Scaffolded Heteroassociative Memory), a novel neocortical-entorhinal-hippocampal network model that achieves high-capacity general associative memory, spatial memory, and episodic memory.

## Core Concepts

The core idea of Vector-HaSH is to use the spatial representations of grid cells as "scaffolds" to organize the associative memory of place cells. This approach extends the neural mechanisms of spatial navigation to general memory storage and retrieval tasks.

## File Overview

### 📊 Main Experimental Files
- `11Rooms_11Maps_Fig_4.ipynb` - Multi-room spatial mapping experiments
- `Scaffold_testing_Fig_2.ipynb` - Scaffold network capacity testing
- `autoencoder_assoc_mem_Fig_3.ipynb` - Autoencoder associative memory comparison
- `Full_model_testing_Fig_3.ipynb` - Complete model testing
- `sequence_autoencoder_Fig_5.ipynb` - Sequence memory experiments
- `Sequence_results_VH_and_baselines_Fig_5.ipynb` - Sequence results vs baselines

### 🔧 Core Code Modules
- `src/assoc_utils_np_2D.py` - 2D associative memory core algorithms
- `src/seq_utils.py` - Sequence learning and processing tools
- `src/capacity_utils.py` - Memory capacity analysis tools

### 📈 Analysis and Visualization
- `MTT.py` - Multiple Trajectory Testing main program
- `Plots_for_baseline_item.py` - Baseline comparison plotting

## 🚀 Quick Start

### Recommended Running Order

**Step 1: Understand Basic Mechanisms**
```bash
# 1. First run scaffold capacity testing to understand basic principles
jupyter notebook Scaffold_testing_Fig_2.ipynb

# 2. Understand associative memory mechanisms
jupyter notebook autoencoder_assoc_mem_Fig_3.ipynb
```

**Step 2: Explore Spatial Capabilities**
```bash
# 3. Multi-room spatial mapping experiments
jupyter notebook 11Rooms_11Maps_Fig_4.ipynb

# 4. Complete model testing
jupyter notebook Full_model_testing_Fig_3.ipynb
```

**Step 3: Sequence and Temporal Dynamics**
```bash
# 5. Sequence memory experiments
jupyter notebook sequence_autoencoder_Fig_5.ipynb

# 6. Sequence results analysis
jupyter notebook Sequence_results_VH_and_baselines_Fig_5.ipynb
```

### Environment Setup
```python
# Main dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import tensorflow as tf  # Only for autoencoder experiments
from tqdm import tqdm
```

### Key Parameter Settings
```python
# Grid cell parameters
lambdas = [3,4,5,7]  # Grid cell module periods
Np = 342             # Number of place cells
Ng = 99              # Number of grid cells (sum of lambdas squared)
thresh = 2.0         # Nonlinearity threshold
c = 0.10             # Connection probability
```

## Detailed File Structure

### Main Experimental Scripts (Jupyter Notebooks)

#### 1. `11Rooms_11Maps_Fig_4.ipynb` - Multi-room Spatial Mapping Experiments
**Function:** Implements experiments from Figure 4, demonstrating the network's spatial representation capabilities across multiple different rooms.
- **2D GC-PC Network Setup:** Establishes 2D grid cell-place cell networks
- **Room Path Generation:** Creates hairpin patterns traversing 10x10 rooms
- **Multi-room Mapping:** Generates 11 different room locations, showcasing spatial remapping abilities
- **Sequence Learning:** Uses MLP classifiers to learn action mappings for path integration
- **Hexagonal Grid Visualization:** Visualizes grid fields and place fields on hexagonal grids
- **Frequency Distribution Analysis:** Analyzes place cell activation patterns across different rooms
- **Inter-room Correlations:** Computes overlap and correlations of place fields between rooms

#### 2. `Scaffold_testing_Fig_2.ipynb` - Scaffold Network Capacity Testing
**Function:** Corresponds to Figure 2, testing the memory capacity of grid cell scaffolds.
- **Grid Codebook Generation:** Uses different ordering strategies (optimal, spiral, hairpin)
- **Capacity Analysis:** Tests memory capacity under different parameters
- **Theoretical Validation:** Verifies theoretical predictions of capacity limits

#### 3. `autoencoder_assoc_mem_Fig_3.ipynb` - Autoencoder Associative Memory Comparison
**Function:** Corresponds to Figure 3, comparing Vector-HaSH with traditional autoencoders.
- **miniImageNet Data Processing:** Loads and preprocesses miniImageNet dataset
- **Autoencoder Implementation:** Implements traditional autoencoders using TensorFlow/Keras
- **Performance Comparison:** Compares reconstruction quality and memory capacity
- **Noise Robustness Testing:** Tests performance under different noise levels

#### 4. `Full_model_testing_Fig_3.ipynb` - Complete Model Testing
**Function:** Tests the complete Vector-HaSH model performance.
- **Complete Network Implementation:** Integrated testing of all components
- **Image Reconstruction:** Tests image associative memory capabilities
- **Baseline Comparisons:** Detailed comparisons with other memory models

#### 5. `sequence_autoencoder_Fig_5.ipynb` & `Sequence_results_VH_and_baselines_Fig_5.ipynb` - Sequence Memory Experiments
**Function:** Corresponds to Figure 5, implementing sequence memory and episodic memory functionality.
- **Sequence Encoding:** Implements encoding and storage of temporal sequences
- **Episodic Memory:** Simulates formation and retrieval of episodic memories
- **Baseline Comparisons:** Comparisons with traditional sequence memory models

#### 6. `VectorHASH_minimal_MLP_seq_Fig_5.ipynb` - Minimal MLP Sequence Model
**Function:** Simplified implementation of sequence learning.

#### 7. `Grid_place_tuning_curves_and_additional_expts_Fig1_4_6.ipynb` - Tuning Curve Analysis
**Function:** Analyzes tuning characteristics of grid cells and place cells.

#### 8. `SplitterCells.ipynb` - Splitter Cell Experiments
**Function:** Studies hippocampal splitter cell properties.

#### 9. `miniimagenet_processing.ipynb` - Data Preprocessing
**Function:** Preprocessing and format conversion of miniImageNet dataset.

### Core Source Code Modules (src/ directory)

#### Associative Memory Tools
- **`assoc_utils.py`**: Basic associative memory utility functions
- **`assoc_utils_np.py`**: NumPy-optimized associative memory implementations
- **`assoc_utils_np_2D.py`**: 2D environment-specific associative memory tools
  - `gen_gbook_2d()`: Generate 2D grid codebooks
  - `path_integration_Wgg_2d()`: 2D path integration matrices
  - `module_wise_NN_2d()`: Modular nearest neighbor search

#### Sequence Processing Tools
- **`seq_utils.py`**: Sequence learning and processing tools
  - `actions()`: Action encoding functions
  - `oneDaction_mapping()`: One-dimensional action mapping
  - Path encoding and decoding functions

#### Sensory and Grid Tools
- **`sensory_utils.py`**: Sensory input processing
- **`sensgrid_utils.py`**: Sensory-grid interactions
- **`sens_pcrec_utils.py`**: Sensory-place cell recurrent connections
- **`sens_sparseproj_utils.py`**: Sparse projection implementations
- **`senstranspose_utils.py`**: Sensory transpose operations

#### Analysis Tools
- **`capacity_utils.py`**: Memory capacity analysis tools
- **`theory_utils.py`**: Theoretical analysis and validation
- **`data_utils.py`**: Data processing utilities

### Main Python Scripts

#### 1. `MTT.py` - Multiple Trajectory Testing
**Function:** Implements multi-trajectory learning and testing functionality.
- **Lesion Experiments:** Simulates effects of hippocampal damage on memory
- **Image Reconstruction:** Tests image reconstruction capabilities under different lesion severities
- **Repeated Learning:** Studies effects of repeated exposure on memory consolidation

#### 2. `MTT_plotting_code.py` - MTT Results Visualization
**Function:** Generates plots and visualizations for MTT experiments.

#### 3. `Plots_for_baseline_item.py` & `Plots_for_baseline_seq.py` - Baseline Comparison Plotting
**Function:** Generates plots comparing with baseline models.
- Standard Hopfield networks
- Pseudo-inverse Hopfield networks
- Sparse connection Hopfield networks
- Autoencoder models

### Data Files

- **`pos420by420.mat`, `pos585by585.mat`, `pos60by60.mat`**: MATLAB format position data containing hexagonal grid position information at different resolutions
- **`paths/xy_coords_1000_2.npy`**: Predefined trajectory coordinate data

## Technical Implementation Details

### Network Architecture
1. **Grid Cell Layer**: Uses periodic encoding to represent spatial positions
2. **Place Cell Layer**: Receives input from grid cells through sparse connections
3. **Sensory Input Layer**: Processes external sensory information
4. **Associative Connections**: Implements memory storage and retrieval

### Key Algorithms
- **Grid Cell Encoding**: Based on multi-module periodic representations
- **Path Integration**: Position updates through matrix multiplication
- **Associative Memory**: Uses pseudo-inverse learning rules
- **Sequence Learning**: Learns action sequences through MLPs

### Data Preprocessing
- **Image Binarization**: Converts color images to binary patterns
- **Noise Injection**: Adds different types and intensities of noise
- **Sequentialization**: Encodes spatial and temporal information into sequences

## Theoretical Contributions

1. **Spatial Scaffold Theory**: First proposal to use grid cell spatial representations as universal memory organization principles
2. **High-Capacity Associative Memory**: Achieves higher storage capacity than traditional Hopfield networks
3. **Multi-Scale Representations**: Implements different scales of spatial representation through multi-module grid encoding
4. **Episodic Memory Mechanisms**: Extends spatial navigation mechanisms to episodic memory storage and retrieval

## Practical Applications

This model provides new insights for the following fields:
- **Neuroscience**: Understanding hippocampal memory mechanisms
- **Artificial Intelligence**: Developing new memory network architectures
- **Cognitive Science**: Explaining relationships between space and memory
- **Machine Learning**: Designing biologically-inspired associative memory systems

## Citation

If you use this code, please cite the original paper:
```
@Article{Chandra&Sharma2025,
author={Chandra, Sarthak and Sharma, Sugandha and Chaudhuri, Rishidev and Fiete, Ila},
title={Episodic and associative memory from spatial scaffolds in the hippocampus},
journal={Nature},
year={2025},
month={Jan},
day={15},
doi={10.1038/s41586-024-08392-y}
}
```

## Contact

For more detailed information, please refer to the original paper: https://www.nature.com/articles/s41586-024-08392-y
