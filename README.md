# Tight-Binding Multilayer Model (Mathematica + Python)

This repository contains two parallel implementations of a multilayer tight-binding model for ferromagnet/normal-metal stacks:

- **`9W.nb`** – the original Mathematica notebook  
- **`multilayer_tb_kubo.py`** – a full Python translation  

Both implementations construct Slater–Koster-style tight-binding Hamiltonians, assemble multilayer stacks (ferromagnet + normal metal), and compute observables such as:

- Band structures along high-symmetry k-paths  
- Density of states (total, layer-resolved, and spin-resolved)  
- Conductivities via the Kubo–Greenwood formula  

The Python version is modular, using `numpy` for linear algebra and exposing hooks to plug in full Slater–Koster tables.

## Features

- Hamiltonian construction: 5 d-orbitals × 2 spins per site, with exchange splitting in ferromagnetic layers.  
- Stack assembly: Arbitrary sequence of F and N layers with interlayer hoppings.  
- Green’s functions: Retarded Green’s function and spectral functions with finite broadening.  
- Observables: DOS, spin/layer-resolved DOS, and conductivity tensors.  
- Band structure: Diagonalization along standard high-symmetry paths for square and triangular lattices.

## Repository Structure

```
.
├── 9W.nb                    # Original Mathematica notebook
├── multilayer_tb_kubo.py    # Python translation (modular, runnable)
├── README.md                # This file
├── requirements.txt         # Python dependencies (suggested)
├── examples/                # Example scripts & plots
│   ├── run_dos.py           # Example: compute DOS and conductivity vs E
│   ├── run_bands.py         # Example: band structure plot
│   └── figures/             # Generated figures
└── LICENSE                  # License file (choose MIT/BSD/GPL/etc.)
```

## Installation (Python)

```bash
git clone https://github.com/yourusername/tb-multilayer.git
cd tb-multilayer
pip install -r requirements.txt
```

## Usage

### Band structure

```python
from multilayer_tb_kubo import example_band_structure
import matplotlib.pyplot as plt

xs, bands = example_band_structure()
plt.plot(xs, bands, 'k-')
plt.show()
```

### DOS and conductivity

```python
from multilayer_tb_kubo import example_compute_dos_and_sigma
import matplotlib.pyplot as plt

Es, dos, sxx = example_compute_dos_and_sigma()
plt.plot(Es, dos, label="DOS")
plt.plot(Es, sxx, label="σxx")
plt.legend()
plt.show()
```

