# Machine learning with NumPy

I am exploring how different machine learning algorithms work by implementing their calculations from scratch with NumPy. The aim is to make each step visible: predictions, the error being measured, and how the model learns from data.

This is a growing learning project. I will add more algorithms and experiments over time, keeping the examples small and easy to follow.

## Approach

- Use Python and NumPy for the algorithm calculations.
- Write the learning steps directly, without ready-made models from machine learning libraries.
- Use small examples to understand the maths and inspect the results.
- Use Matplotlib only to draw results; it does not perform the learning.

## Algorithms and experiments

| Item | What it explores | Code |
| --- | --- | --- |
| Linear regression with gradient descent | Fit a line by adjusting its slope and intercept, measure mean squared error, and stop when the change in error is small. | [src/gadient_descent.py](src/gadient_descent.py) |

New algorithms and experiments will be added to this list as they are implemented.

## Run the current example

Requires Python 3, NumPy, and Matplotlib.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install numpy matplotlib
python src/gadient_descent.py
```

The script uses six illustrative pairs of study hours and test scores. It prints the training error, the final slope and intercept, and displays a plot of the data with the fitted line. These values are a small learning example, not evidence about real student performance.

To run without opening a plot window:

```bash
MPLBACKEND=Agg python src/gadient_descent.py
```

## Project notes

The current example uses a fixed learning rate, an iteration limit, and a stopping tolerance based on the change in mean squared error. More explanations, comparisons, and algorithms can be added alongside the code as the project grows.

[Read the portfolio entry](https://ruddirodriguez.github.io/numpy-ml.html)
