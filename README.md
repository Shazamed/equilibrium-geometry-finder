# chem-equilibrium-sim
Calculate the equilibrium geometry of a random distribution of atoms, each interacting with a set potential.
## Description
The script calculates and attempts to finds the lowest energy geometry and writes the coordinates to an .xyz file.

The number of atoms, runs, and potential used can be specified in the command line interface. The lowest energy geometry will be saved in output.xyz. The table of interparticle distances are also saved in table.out.

Some precalculated equilibrium geometries are stored in the Lennard-Jones-geometries or Morse-X-geometries directory in this repository. The lowest energy conformation is the bipentagonal geometry which are saved in the bipentagonal-XX.xyz file in the respective directories.
## Usage
Navigate to the directory where the repository is cloned.

Simply run main.py with python:
```
python main.py
```