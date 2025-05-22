# Numerical Methods Graphing Tool

A Python application for visualizing and solving equations using various numerical methods.

## Features
- Graphical visualization of functions
- Multiple numerical methods for root finding:
  - Bisection Method
  - Secant Method
  - Newton-Raphson Method
  - Regula Falsi Method
  - Incremental Method
- Interactive GUI with draggable annotations
- Support for both equation and data point inputs
- System of equations solver

## Requirements
- Python 3.7 or higher
- Required packages listed in `requirements.txt`

## Installation
1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/graphing.git
cd graphing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run the application:
```bash
python graph.py
```

### Input Format
- Enter equations using Python syntax
- Examples:
  - `x**3 - x - 2`
  - `sqrt(x)`
  - `sin(x)`
  - `exp(x)`

### Features
- Set lower and upper limits for the graph
- Adjust step size for calculations
- Choose between different numerical methods
- View detailed iteration tables
- Drag and reposition annotations

## License
MIT License 