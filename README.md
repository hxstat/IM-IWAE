# IM-IWAE: Identifiable Deep Generative Models for MNAR Data

This repository contains the implementation of **IM-IWAE** (Identifiable Missing-not-at-random Importance-Weighted
AutoEncoder), a deep generative model designed to handle data with missingness that is not at random (MNAR), under the condition of no self-censoring given latent variables for identifiability.

The core implementation is in `IM-IWAE.py`.

## Quick Start


### Installation

Download the code or clone this repository by running:

```bash
git clone https://github.com/your-username/IM-IWAE.git
cd IM-IWAE
```

### Prerequisites

Before you begin, ensure you have the following installed:
*   **Python 3.8.5** (Recommended for compatibility)
*   To install the required Python packages, run:
    ```bash
    pip install -r requirements.txt
    ```
    
## Experiments

This repository contains code to reproduce the synthetic and real-world experiments from the paper.

### Synthetic Data Experiments

These experiments demonstrate the model's performance on synthetic data.

*   **3-Dimensional Simulation**
    *   Run the script: `simulation/sim3d.py`
*   **Mixture of Gaussians Simulation**
    *   Run the script: `simulation/mixgaussian.py`

### Real-World Data Experiments

These experiments evaluate the model on real-world datasets, with simualted MNAR mechanisms for the UCI datasets and inherent MNAR mechanisms for the other datasets.

*   **UCI Datasets**
    *   Run the script: `real_data_analysis/uci_mnar.py`
*   **HIV Data (Binary 3D)**
    *   Run the script: `real_data_analysis/binary3d.py`
*   **Yahoo! R3 Music Ratings**
    *   Run the script: `real_data_analysis/music_ratings.py`

### Important Note on Datasets

**The real-world datasets are not included in this repository.** You will need to download them from their original sources before running the respective scripts. 

**Exception:** The Wine Quality dataset URL is included in `uci_mnar.py`, so the script for this dataset can be executed directly without manual download.
