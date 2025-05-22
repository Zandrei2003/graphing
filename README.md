# Numerical Methods Root Finder

A Python application that visualizes and solves equations using various numerical methods for finding roots. Built with Tkinter and Matplotlib.

## Features

- Multiple root-finding methods:
  - Bisection Method
  - Secant Method
  - Newton-Raphson Method
  - Regula Falsi Method
  - Incremental Method
  - Graphical Method
- Interactive visualization of the solution process
- Support for both equation and data point inputs
- Draggable annotations
- System of equations solver
- Beautiful dark theme UI

## Requirements

```bash
pip install numpy matplotlib sympy scipy ttkthemes
```

## Usage

1. Run the application:
```bash
python graph.py
```

2. Enter your equation in the input box (e.g., `x**3 - x - 2`)
3. Set the lower and upper limits for the search interval
4. Choose a method from the sidebar
5. View the results in the graph and table

## Examples

- `x**3 - x - 2`
- `sqrt(x) - 2`
- `sin(x) - 0.5`
- `exp(x) - 2`

## License

MIT License 