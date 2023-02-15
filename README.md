# GA to GA

A meta-learning paradigm that applies genetic algorithms to optimize the hyperparameters of genetic algorithms.

The "child" GA solves TSP from the TSPLIB dataset in this case, but it can also solve any other problem.

Team of three: SW Li, PY Tsai, YS Liao

## Usage

```bash
python Parent_GA.py
```

## File Description

### Parent_GA.py

A parameter optimizer for Child_GA_TSP.py using genetic algorithm

### Child_GA_TSP.py

A TSP solver using genetic algorithm, wrapped with a configurable hyperparameter setting.

### load_tsp.py

Utility for loading TSP maps. TSP maps are from TSPLIB dataset.

### Report.pdf

A Chinese report including methodology and experiment results.

### Slide.pdf

A 10-minute presentation slide.
