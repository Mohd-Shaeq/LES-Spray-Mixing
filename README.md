# LES-Spray-Mixing
LES-based hydrogen spray mixing workflow using OpenFOAM snappyHexMesh and Python post-processing for quantitative mixing metrics.
# LES Spray Mixing – Meshing & Post-Processing

This repository contains the numerical setup and post-processing workflow
developed for a Master’s thesis on LES-based hydrogen spray mixing
using OpenFOAM®.

---

## 1. Mesh Generation (snappyHexMesh)

The computational mesh is generated using `snappyHexMesh` (OpenFOAM v2112).

### Key characteristics
- Hexahedral-dominant mesh
- Explicit feature edge refinement
- Distance-based volumetric refinement along injector axis
- Boundary layer meshing intentionally disabled

### Files
- `meshing/snappyHexMeshDict`
- `meshing/Geometrie.stl`
- `meshing/Geometrie.eMesh`

---

## 2. Post-Processing: Mixing Metrics (Python)

The Python script computes quantitative hydrogen–air mixing metrics from
OpenFOAM field data.

### Computed metrics
- Mixture fraction variance (KMV)
- Mixing index
- Segregation index
- Rich / stoichiometric / lean zone fractions (λ-based)

### File
- `postprocessing/mixing_metrics.py`

---

## 3. Reproducibility

### Requirements
- OpenFOAM v2112
- Python 3 with NumPy and Matplotlib

### Meshing
```bash
blockMesh
snappyHexMesh -overwrite
