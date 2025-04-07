# Solving QUBO using SPVAR

### What is QUBO
NP-hard task:

Given matrix $Q \in R^{n \times n}$. Find $x \in {0, 1}^{n}$ that minimizes $x^TQx$.

Common solution - annealing.

### What is SPVAR

Optimization algirithm for annealing. Viewed simply, might be described like this: generated many samples from solver, calculates some statistic on them, reduce task size ($n$) and solve simplified task.

[Full article](https://arxiv.org/abs/1606.07797)

### Results of research

Algorithm works best on logistic tasks: SPVAR reduces the task size by 80% and its solutions are better by 4% compares to simply annealing solutions.

[Full results presentation](Presentation.pdf)
