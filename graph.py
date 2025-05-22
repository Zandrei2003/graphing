import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
from scipy.interpolate import interp1d
from ttkthemes import ThemedTk

# --- Draggable Annotation Helper ---
class DraggableAnnotation:
    def __init__(self, annotation):
        self.annotation = annotation
        self.press = None
        self.background = None
        self.cidpress = annotation.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = annotation.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = annotation.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        contains, attrd = self.annotation.contains(event)
        if not contains: return
        self.press = (self.annotation.xyann, event.xdata, event.ydata)

    def on_motion(self, event):
        if self.press is None: return
        if event.xdata is None or event.ydata is None: return
        xyann, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        new_xy = (xyann[0] + dx, xyann[1] + dy)
        self.annotation.set_position(new_xy)
        self.annotation.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.annotation.figure.canvas.draw_idle()

def preprocess_equation(eq):
    eq = eq.strip()
    # If equation contains '=', move all terms to one side (handle multiple '=')
    if eq.count('=') > 1:
        # Split at the last '='
        last_eq = eq.rfind('=')
        left = eq[:last_eq]
        right = eq[last_eq+1:]
    elif '=' in eq:
        left, right = eq.split('=', 1)
    else:
        left, right = eq, ''
    # Remove leading f(x)=, y=, or similar from left
    left = re.sub(r'^[fF]\(x\)\s*=\s*', '', left)
    left = re.sub(r'^y\s*=\s*', '', left)
    # Remove '= 0' or '=0' at the end of right
    right = right.strip()
    if right == '0':
        eq = left
    elif right:
        eq = f"({left})-({right})"
    else:
        eq = left
    # Remove '= 0' or '=0' at the end (again, for safety)
    eq = re.sub(r'=\s*0\s*$', '', eq)
    # Replace ^ with **
    eq = eq.replace('^', '**')
    # Replace unicode sqrt with space and parentheses: '√ (x)' or '√(x)'
    eq = re.sub(r'√\s*\(\s*([^)]+)\s*\)', r'sqrt(\1)', eq)
    # Replace unicode sqrt and abs
    eq = re.sub(r'√\s*([a-zA-Z0-9_()]+)', r'sqrt(\1)', eq)
    eq = re.sub(r'√\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?', r'sqrt(\1)', eq)
    eq = re.sub(r'\|([a-zA-Z0-9_()]+)\|', r'abs(\1)', eq)
    # --- New: Convert log/ln <expr> to log(<expr>) for any valid expression ---
    # This must come before the other log/ln regexes
    # def log_ln_expr_repl(m):
    #     func = m.group(1)
    #     expr = m.group(2)
    #     return f'{func}({expr})'
    # Only match if not already followed by a parenthesis
    # eq = re.sub(
    #     r'(log|ln)\s+(?!\()([a-zA-Z0-9_]+(?:\s*[+\-*/^]\s*[a-zA-Z0-9_()]+)*)',
    #     log_ln_expr_repl,
    #     eq
    # )
    # --- Bulletproof ln/log handling (do this FIRST) ---
    # ln(x) -> log(x), lnx -> log(x), ln x -> log(x), logx -> log(x), log x -> log(x)
    eq = re.sub(r'ln\s*\(\s*([^)]+)\s*\)', r'log(\1)', eq)  # ln(x) -> log(x)
    eq = re.sub(r'ln\s*([a-zA-Z][a-zA-Z0-9_]*)', r'log(\1)', eq)  # lnx or ln x -> log(x)
    eq = re.sub(r'log\s*\(\s*([^)]+)\s*\)', r'log(\1)', eq)  # log(x) -> log(x) (idempotent)
    # PATCH: Only match log x if not already followed by '(' (negative lookahead)
    eq = re.sub(
        r'log\s+(?!\()([a-zA-Z][a-zA-Z0-9_]*)',
        r'log(\1)',
        eq
    )  # log x -> log(x), but not log(...)
    # Remove any accidental log*x (function times variable) to log(x)
    eq = re.sub(r'log\s*\*\s*([a-zA-Z][a-zA-Z0-9_]*)', r'log(\1)', eq)
    # --- End ln/log handling ---
    # Replace e^x with exp(x)
    eq = re.sub(r'e\s*\*\*\s*([a-zA-Z0-9_()]+)', r'exp(\1)', eq)
    eq = re.sub(r'e\s*\^\s*([a-zA-Z0-9_()]+)', r'exp(\1)', eq)
    # Replace np. functions with plain equivalents
    eq = re.sub(r'np\s*\.\s*sin', 'sin', eq, flags=re.IGNORECASE)
    eq = re.sub(r'np\s*\.\s*cos', 'cos', eq, flags=re.IGNORECASE)
    eq = re.sub(r'np\s*\.\s*tan', 'tan', eq, flags=re.IGNORECASE)
    eq = re.sub(r'np\s*\.\s*exp', 'exp', eq, flags=re.IGNORECASE)
    eq = re.sub(r'np\s*\.\s*log', 'log', eq, flags=re.IGNORECASE)
    eq = re.sub(r'np\s*\.\s*sqrt', 'sqrt', eq, flags=re.IGNORECASE)
    eq = re.sub(r'np\s*\.\s*abs', 'abs', eq, flags=re.IGNORECASE)
    # Convert sinx, cosx, etc. to sin(x)
    eq = re.sub(
        r'(sin|cos|tan|exp|log|sqrt|abs)\s*([a-zA-Z][a-zA-Z0-9_]*)',
        r'\1(\2)', eq)
    # Convert 'sin x' to 'sin(x)', etc.
    eq = re.sub(
        r'(sin|cos|tan|exp|log|sqrt|abs)\s*\(\s*([^)]+)\s*\)',
        r'\1(\2)', eq)
    # Insert * between number and variable/function (e.g., 2x, 2 sin(x))
    eq = re.sub(r'(\d)\s*([a-zA-Z(])', r'\1*\2', eq)
    # Insert * between variable and number (e.g., x2 -> x*2)
    eq = re.sub(r'([a-zA-Z])([0-9])', r'\1*\2', eq)
    # Insert * between digit and (
    eq = re.sub(r'(\d)\(', r'\1*(', eq)
    # Insert * between variable and (
    eq = insert_mul_var_paren(eq)
    # Insert * between ) and (
    eq = re.sub(r'\)\(', r')*(', eq)
    # Remove any trailing non-matching characters
    eq = re.sub(r'[^0-9a-zA-Z)]+$', '', eq)
    # Remove trailing unmatched parentheses
    while eq.endswith(')') and eq.count('(') < eq.count(')'):
        eq = eq[:-1]
    # Check for balanced parentheses
    if eq.count('(') != eq.count(')'):
        raise ValueError(
            "Unbalanced parentheses in the equation. Please check your input."
        )
    # Ensure all log uses are function calls: log x -> log(x), log*x -> log(x)
    eq = re.sub(r'log\s*\*\s*([a-zA-Z][a-zA-Z0-9_]*)', r'log(\1)', eq)
    eq = re.sub(r'log\s+([a-zA-Z][a-zA-Z0-9_]*)', r'log(\1)', eq)
    # Final fix: convert any log*x, log * x, log *(x), log * (x) to log(x)
    eq = re.sub(r'log\s*\*\s*\(?\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\)?', r'log(\1)', eq)
    # Also handle log x (with a space) at the end
    eq = re.sub(r'log\s+([a-zA-Z][a-zA-Z0-9_]*)', r'log(\1)', eq)
    # Final fix: convert any sqrt*x, sqrt * x, sqrt *(x), sqrt * (x) to sqrt(x)
    eq = re.sub(r'sqrt\s*\*\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?', r'sqrt(\1)', eq)
    # Also handle sqrt x (with a space) at the end
    eq = re.sub(r'sqrt\s+([a-zA-Z0-9_]+)', r'sqrt(\1)', eq)
    # Replace e^... or e**... with exp(...), robustly handling negative exponents and parentheses
    # Handles e^-x, e^(-x), e**-x, e**(-x), e^(x+1), etc.
    eq = re.sub(
        r'e\s*\*\*\s*(-?\s*\(?[a-zA-Z0-9_+\-*/ ]+\)?)',
        r'exp(\1)', eq
    )
    eq = re.sub(
        r'e\s*\^\s*(-?\s*\(?[a-zA-Z0-9_+\-*/ ]+\)?)',
        r'exp(\1)', eq
    )
    print('DEBUG: Preprocessed equation:', eq)
    return eq

def insert_mul_var_paren(eq):
    # Insert * between a single variable and ( but not after function names
    # Only match a single letter variable not preceded by another letter (i.e., not part of a function name)
    return re.sub(r'(?<![a-zA-Z])([a-zA-Z])\(', r'\1*(', eq)

def secant_method(f_lambda, x0, x1, tol=1e-5, max_iter=50):
    """Performs the Secant Method for root finding."""
    iterations = []
    for _ in range(max_iter):
        f_x0 = f_lambda(x0)
        f_x1 = f_lambda(x1)
        if abs(f_x1 - f_x0) < 1e-12:
            break  # Avoid division by zero
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        iterations.append((x0, x1, x2))
        if abs(x2 - x1) < tol:
            return x2, iterations
        x0, x1 = x1, x2
    return x2, iterations


def newton_raphson_method(f_lambda, df_lambda, x0, tol=1e-8, max_iter=100):
    """Performs the Newton-Raphson Method for root finding."""
    iterations = []
    for _ in range(max_iter):
        f_x0 = f_lambda(x0)
        df_x0 = df_lambda(x0)
        if abs(df_x0) < 1e-12:
            break  # Avoid division by zero
        x1 = x0 - f_x0 / df_x0
        iterations.append((x0, x1))
        if abs(x1 - x0) < tol:
            return x1, iterations
        x0 = x1
    return x1, iterations


def regula_falsi_method(f_lambda, a, b, tol=1e-5, max_iter=50):
    """Performs the Regula Falsi (False Position) Method for root finding."""
    if f_lambda(a) * f_lambda(b) >= 0:
        return None, []  # No root in this interval
    iterations = []
    for _ in range(max_iter):
        fa = f_lambda(a)
        fb = f_lambda(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f_lambda(c)
        iterations.append((a, b, c))
        if abs(fc) < tol:
            return c, iterations
        if fa * fc < 0:
            b = c  # Root is in left half
        else:
            a = c  # Root is in right half
    return c, iterations

# --- Bisection Method Implementation ---
def bisection_method(f_lambda, a, b, tol=1e-5, max_iter=50):
    """Performs the Bisection Method for root finding."""
    if f_lambda(a) * f_lambda(b) >= 0:
        return None, []  # No root in this interval
    iterations = []
    for _ in range(max_iter):
        c = (a + b) / 2.0
        fa = f_lambda(a)
        fc = f_lambda(c)
        iterations.append((a, b, c))
        if abs(fc) < tol or abs(b - a) / 2 < tol:
            return c, iterations
        if fa * fc < 0:
            b = c
        else:
            a = c
    return c, iterations

# --- Run Bisection Method and Plot ---
def run_bisection_method():
    try:
        global draggable_annotations
        draggable_annotations.clear()
        eq = preprocess_equation(equation_entry.get())
        x_sym = sp.symbols('x')
        f_sympy = sp.sympify(eq, locals={'abs': sp.Abs, 'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        f_lambda = sp.lambdify(x_sym, f_sympy, 'numpy')

        a, b, step = get_limits()
        tol = 1e-7
        max_iter = 100
        roots = []
        all_iterations = []
        intervals = []
        x_scan = np.arange(a, b, step)
        for i in range(len(x_scan) - 1):
            x0, x1 = x_scan[i], x_scan[i+1]
            try:
                f0, f1 = f_lambda(x0), f_lambda(x1)
                f0 = np.asarray(f0, dtype=float)
                f1 = np.asarray(f1, dtype=float)
                if np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                    continue
                if f0 * f1 < 0:
                    root, iterations = bisection_method(f_lambda, x0, x1, tol=tol, max_iter=max_iter)
                    if root is not None:
                        if not any(abs(root - r) < tol*10 for r in roots):
                            roots.append(root)
                            all_iterations.append(iterations)
                            intervals.append((x0, x1))
            except Exception:
                continue
        # Plotting
        x_values = np.linspace(a, b, 1000)
        y_values = f_lambda(x_values)
        ax.clear()
        plt.style.use('ggplot')
        dx = (b - a) * 0.04
        dy = (max(y_values) - min(y_values)) * 0.08
        
        # Plot the main function line first
        ax.plot(x_values, y_values, label=f"f(x) = {eq}", color='white', linewidth=2.5)
        
        for idx, (root, iterations) in enumerate(zip(roots, all_iterations)):
            # Plot bisection intervals with improved visibility
            for i, (a_i, b_i, c_i) in enumerate(iterations):
                ax.plot([a_i, b_i], [f_lambda(a_i), f_lambda(b_i)], 
                       color='#ff9800',  # Brighter orange
                       linestyle='-', 
                       linewidth=2.5,  # Thicker lines
                       marker='o', 
                       markersize=6,  # Larger markers
                       alpha=0.7 if i < len(iterations)-1 else 1.0,  # More opaque for final iteration
                       label="Bisection Interval" if i == 0 and idx == 0 else "")
                ax.scatter(c_i, f_lambda(c_i), 
                          color='#ff5722',  # Deep orange
                          s=50,  # Larger size
                          alpha=0.9,
                          edgecolors='white',  # White border
                          linewidth=1.5)
                # Add iteration number labels
                if i < len(iterations)-1:  # Don't label the final iteration
                    ann = ax.annotate(f"Iter {i+1}", 
                              xy=(c_i, f_lambda(c_i)),
                              xytext=(c_i + dx*0.5, f_lambda(c_i) + dy*0.5),
                              color='#ff9800',
                              fontsize=10,
                              fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="#ff9800", lw=1, alpha=0.8),
                              arrowprops=dict(arrowstyle="->", color='#ff9800', lw=1.5))
                    draggable_annotations.append(DraggableAnnotation(ann))
            ax.scatter(root, f_lambda(root), 
                      color='lime', 
                      s=150,  # Larger size
                      edgecolors='black', 
                      linewidth=2,  # Thicker border
                      zorder=6, 
                      label=f"Bisection Root ≈ {root:.7f}" if idx == 0 else None)
            ax.axvline(root, 
                      linestyle="dashed", 
                      color="lime", 
                      linewidth=2,  # Thicker line
                      alpha=0.8)
            # Improved root annotation
            stagger = (idx % 5) * 18
            ann = ax.annotate(f"Bisection Root ≈ {root:.7f}",
                       xy=(root, f_lambda(root)),
                       xytext=(root, f_lambda(root) + dy + stagger),
                       color='lime', 
                       fontsize=12,  # Larger font
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", fc="#23272e", ec="lime", lw=2, alpha=0.9),
                       arrowprops=dict(arrowstyle="->", color='lime', lw=2),
                       ha='center', 
                       va='bottom')
            draggable_annotations.append(DraggableAnnotation(ann))
        # After plotting the function line
        x_annot = x_values[int(len(x_values) * 0.8)]
        y_annot = y_values[int(len(y_values) * 0.8)]
        ann = ax.annotate(
            f"f(x) = {eq}",
            xy=(x_annot, y_annot),
            xytext=(x_annot + 1, y_annot + (max(y_values) - min(y_values)) * 0.1),
            color='white',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="white", lw=1, alpha=0.8),
            arrowprops=dict(arrowstyle="->", color='white', lw=1),
            ha='left', va='center'
        )
        draggable_annotations.append(DraggableAnnotation(ann))
        # Grid and axes
        ax.axhline(0, color='white', linewidth=1.5)
        ax.axvline(0, color='white', linewidth=1.5)
        ax.grid(True, linestyle='--', linewidth=0.8, color='gray', alpha=0.7)
        ax.set_xlabel("X-Axis", color='white', fontsize=12)
        ax.set_ylabel("Y-Axis", color='white', fontsize=12)
        ax.set_title(f"Bisection Method: {eq}", color='white', fontsize=14)
        ax.legend(loc="upper left", fontsize=10)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        canvas.draw()
        if roots:
            root_display.set_root_text("BISECTION ROOTS: " + ", ".join([f"x ≈ {r:.7f}" for r in roots]))
        else:
            root_display.set_root_text("NO ROOTS FOUND")
        add_tip_annotation(ax, draggable_annotations)
    except Exception as e:
        messagebox.showerror("Error", f"Bisection Method Error: {e}")

def get_limits():
    # Helper to get lower, upper, step from UI, with defaults and validation
    try:
        a = float(lower_entry.get())
    except Exception:
        a = -10
    try:
        b = float(upper_entry.get())
    except Exception:
        b = 10
    try:
        step = float(step_entry.get())
        if step <= 0:
            step = 0.1
    except Exception:
        step = 0.1
    if a >= b:
        a, b = -10, 10
    return a, b, step

def get_function_from_input():
    """
    Returns a tuple (f_lambda, x_range, label) depending on the input mode.
    - For equation mode: returns (f_lambda, x_range, label)
    - For data points mode: parses data points, returns interpolated function,
      x_range, and label
    """
    if mode_var.get() == 'equation':
        eq = preprocess_equation(equation_entry.get())
        x_sym = sp.symbols('x')
        f_sympy = sp.sympify(eq, locals={'abs': sp.Abs, 'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        # Check if the function depends on x
        if x_sym not in f_sympy.free_symbols:
            raise ValueError(
                "The function does not depend on x. Please enter a function of x."
            )
        f_lambda = sp.lambdify(x_sym, f_sympy, 'numpy')

        a, b, step = get_limits()
        x_range = np.arange(a, b, step)
        if len(x_range) < 2:
            x_range = np.linspace(a, b, 1000)
        label = f"f(x) = {eq}"
        return f_lambda, x_range, label
    else:
        # Parse data points from text box
        lines = data_points_text.get("1.0", tk.END).strip().split('\n')
        x_vals, y_vals = [], []
        for line in lines:
            if ',' in line:
                try:
                    x, y = map(float, line.split(','))
                    x_vals.append(x)
                    y_vals.append(y)
                except Exception:
                    continue
        if len(x_vals) < 2:
            raise ValueError("At least two data points are required.")
        # Sort by x
        x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals)))
        f_interp = interp1d(x_vals, y_vals, kind='cubic', fill_value="extrapolate")
        x_range = np.linspace(min(x_vals), max(x_vals), 1000)
        label = "Interpolated Data Points"
        return f_interp, x_range, label

def run_graphical_method():
    try:
        global draggable_annotations
        draggable_annotations.clear()
        f_lambda, x_values, label = get_function_from_input()
        ax.clear()
        plt.style.use('ggplot')
        y_values = f_lambda(x_values)
        ax.plot(x_values, y_values, label=label, color='white', linewidth=2)
        # Highlight x-axis
        ax.axhline(0, color='lime', linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(0, color='white', linewidth=1.2)
        ax.grid(True, linestyle='--', linewidth=0.7, color='gray')
        ax.set_xlabel("X-Axis", color='white', fontsize=12)
        ax.set_ylabel("Y-Axis", color='white', fontsize=12)
        ax.set_title(f"Graphical Method: {label}", color='white', fontsize=14)
        ax.legend(loc="upper left", fontsize=10)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        # Add draggable annotation for demonstration
        x_annot = x_values[int(len(x_values) * 0.8)]
        y_annot = y_values[int(len(y_values) * 0.8)]
        ann = ax.annotate(
            f"f(x) = {label}",
            xy=(x_annot, y_annot),
            xytext=(x_annot + 1, y_annot + (max(y_values) - min(y_values)) * 0.1),
            color='white',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="white", lw=1, alpha=0.8),
            arrowprops=dict(arrowstyle="->", color='white', lw=1),
            ha='left', va='center'
        )
        draggable_annotations.append(DraggableAnnotation(ann))
        canvas.draw()
        root_display.set_root_text(
            "Graphical method: visually inspect where the curve "
            "crosses the x-axis."
        )
        add_tip_annotation(ax, draggable_annotations)
    except Exception as e:
        messagebox.showerror("Error", f"Graphical Method Error: {e}")

def run_incremental_method():
    try:
        global draggable_annotations
        draggable_annotations.clear()
        f_lambda, x_values, label = get_function_from_input()
        a, b, step = get_limits()
        step = 0.05  # More accurate step
        intervals = []
        x_scan = np.arange(a, b, step)
        for i in range(len(x_scan) - 1):
            x0, x1 = x_scan[i], x_scan[i+1]
            try:
                f0, f1 = f_lambda(x0), f_lambda(x1)
                f0 = np.asarray(f0, dtype=float)
                f1 = np.asarray(f1, dtype=float)
                if np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                    continue
                if f0 * f1 < 0:
                    intervals.append((x0, x1))
            except Exception:
                continue
        # Plotting
        ax.clear()
        plt.style.use('ggplot')
        y_values = f_lambda(x_values)
        ax.plot(x_values, y_values, label=label, color='white', linewidth=2)
        dx = (b - a) * 0.04
        dy = (max(y_values) - min(y_values)) * 0.08
        for idx, (x0, x1) in enumerate(intervals):
            label_bracket = 'Root Bracket' if idx == 0 else None
            ax.axvspan(
                x0, x1, color='orange', alpha=0.3,
                label=label_bracket
            )
            # Annotate bracket with pointer, further away
            mid_x = (x0 + x1) / 2
            mid_y = (f_lambda(x0) + f_lambda(x1)) / 2
            stagger = (idx % 5) * dy * 2  # Change 5 to a higher number if you expect more brackets
            ann = ax.annotate(
                f"Bracket [{x0:.2f}, {x1:.2f}]",
                xy=(mid_x, mid_y),
                xytext=(mid_x + dx, mid_y + dy + stagger),
                color='orange',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="orange", lw=1, alpha=0.8),
                arrowprops=dict(arrowstyle="->", color='orange', lw=2),
                ha='left', va='bottom'
            )
            draggable_annotations.append(DraggableAnnotation(ann))
        ax.axhline(0, color='white', linewidth=1.2)
        ax.axvline(0, color='white', linewidth=1.2)
        ax.grid(True, linestyle='--', linewidth=0.7, color='gray')
        ax.set_xlabel("X-Axis", color='white', fontsize=12)
        ax.set_ylabel("Y-Axis", color='white', fontsize=12)
        ax.set_title(
            f"Incremental Method: {label}", color='white', fontsize=14
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        canvas.draw()
        if intervals:
            interval_str = ", ".join([
                f"[{x0:.2f}, {x1:.2f}]" for x0, x1 in intervals
            ])
            root_display.set_root_text(
                "INCREMENTAL: Bracketing Intervals: "
                + interval_str
            )
        else:
            root_display.set_root_text("INCREMENTAL: No bracketing intervals found. ")
        add_tip_annotation(ax, draggable_annotations)
    except Exception as e:
        messagebox.showerror("Error", f"Incremental Method Error: {e} ")

def generate_graph():
    try:
        global draggable_annotations
        draggable_annotations.clear()
        f_lambda, x_values, label = get_function_from_input()
        # Use user-defined interval and step size
        a, b, step = get_limits()
        tol = 1e-5
        roots = []
        all_iterations = []
        intervals = []
        x_scan = np.arange(a, b, step)
        for i in range(len(x_scan) - 1):
            x0, x1 = x_scan[i], x_scan[i+1]
            try:
                f0, f1 = f_lambda(x0), f_lambda(x1)
                f0 = np.asarray(f0, dtype=float)
                f1 = np.asarray(f1, dtype=float)
                if np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                    continue
                if f0 * f1 < 0:
                    # Bisection method removed: just record the interval
                    if not any(abs((x0 + x1)/2 - r) < tol*10 for r in roots):
                        roots.append((x0 + x1)/2)
                        intervals.append((x0, x1))
            except Exception:
                continue
        # Plotting
        ax.clear()
        plt.style.use('ggplot')
        y_values = f_lambda(x_values)
        ax.plot(x_values, y_values, label=label, color='white', linewidth=2)
        # Plot all roots and their intervals
        for idx, root in enumerate(roots):
            ax.scatter(
                root, f_lambda(root), color='lime', s=100, edgecolors='black',
                label=f"Root ≈ {root:.5f}" if idx == 0 else None
            )
            ax.axvline(root, linestyle="dashed", color="lime", linewidth=1)
            ann = ax.annotate(
                f"x ≈ {root:.5f}",
                (root, f_lambda(root)),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='lime',
                fontsize=12,
                fontweight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="#23272e",
                    ec="lime",
                    lw=1,
                    alpha=0.8
                )
            )
            draggable_annotations.append(DraggableAnnotation(ann))
        # Grid, labels, and aesthetics
        ax.axhline(0, color='white', linewidth=1.2)
        ax.axvline(0, color='white', linewidth=1.2)
        ax.grid(True, linestyle='--', linewidth=0.7, color='gray')
        ax.set_xlabel("X-Axis", color='white', fontsize=12)
        ax.set_ylabel("Y-Axis", color='white', fontsize=12)
        ax.set_title(
            f"Numerical Methods Graph: {label}",
            color='white', fontsize=14
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        canvas.draw()
        # Update root label
        if roots:
            root_display.set_root_text("ROOTS FOUND: " + ", ".join([f"x ≈ {r:.6f}" for r in roots]))
        else:
            root_display.set_root_text("NO ROOTS FOUND")
        add_tip_annotation(ax, draggable_annotations)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input! {e}")

def generate_table():
    try:
        eq = preprocess_equation(equation_entry.get())
        x_sym = sp.symbols('x')
        f_sympy = sp.sympify(eq, locals={'abs': sp.Abs, 'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        f_lambda = sp.lambdify(x_sym, f_sympy, 'numpy')

        a, b, step = get_limits()

        for i in table.get_children():
            table.delete(i)

        iteration = 1
        while a <= b:
            fa = f_lambda(a)
            fb = f_lambda(a + step)
            remark = "Go to next interval" if fa * fb > 0 else "Revert back & consider smaller interval"
            table.insert("", "end", values=(iteration, f"{a:.3f}", f"{step:.3f}", f"{a + step:.3f}", f"{fa:.6f}", f"{fb:.6f}", f"{fa * fb:.6f}", remark))
            a += step
            iteration += 1
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input! {e}")

def find_root():
    try:
        f_lambda, x_values, label = get_function_from_input()
        a, b, step = get_limits()
        tol = 1e-5
        roots = []
        x_scan = np.arange(a, b, step)
        for i in range(len(x_scan) - 1):
            x0, x1 = x_scan[i], x_scan[i+1]
            try:
                f0, f1 = f_lambda(x0), f_lambda(x1)
                f0 = np.asarray(f0, dtype=float)
                f1 = np.asarray(f1, dtype=float)
                if np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                    continue
                if f0 * f1 < 0:
                    # Bisection method removed: just record the midpoint as root
                    if not any(abs((x0 + x1)/2 - r) < tol*10 for r in roots):
                        roots.append((x0 + x1)/2)
            except Exception:
                continue
        if roots:
            root_display.set_root_text("Root found = " + ", ".join([f"{r:.6f}" for r in roots]))
        else:
            root_display.set_root_text("No root found")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input! {e}")

def run_secant_method():
    try:
        global draggable_annotations
        draggable_annotations.clear()
        eq = preprocess_equation(equation_entry.get())
        x_sym = sp.symbols('x')
        f_sympy = sp.sympify(eq, locals={'abs': sp.Abs, 'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        f_lambda = sp.lambdify(x_sym, f_sympy, 'numpy')

        a, b, step = get_limits()
        tol = 1e-8   # Tighter tolerance for accuracy
        max_iter = 100
        roots = []
        all_iterations = []
        x_scan = np.arange(a, b, step)
        for i in range(len(x_scan) - 1):
            x0, x1 = x_scan[i], x_scan[i+1]
            try:
                f0, f1 = f_lambda(x0), f_lambda(x1)
                f0 = np.asarray(f0, dtype=float)
                f1 = np.asarray(f1, dtype=float)
                if np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                    continue
                if f0 * f1 < 0:  # Only if sign changes (root is bracketed)
                    try:
                        root, iterations = secant_method(f_lambda, x0, x1, tol=tol, max_iter=max_iter)
                        if root is not None and np.isfinite(root):
                            if a - step <= root <= b + step:
                                if not any(abs(root - r) < tol*10 for r in roots):
                                    roots.append(root)
                                    all_iterations.append(iterations)
                    except Exception:
                        continue
            except Exception:
                continue
        # Plotting
        x_values = np.linspace(a, b, 1000)
        y_values = f_lambda(x_values)
        ax.clear()
        ax.plot(x_values, y_values, label=f"f(x) = {eq}", color='white', linewidth=2)
        # Plot secant lines and roots (only for successful root-finding sequences)
        secant_line_label_shown = False
        root_label_shown = False
        for idx, (root, iterations) in enumerate(zip(roots, all_iterations)):
            if len(iterations) > 0:
                # Only plot the sequence for the successful root
                for i, (x0_i, x1_i, x2_i) in enumerate(iterations):
                    ax.plot([x0_i, x1_i], [f_lambda(x0_i), f_lambda(x1_i)], color='#00bfff', linestyle='-', marker='o', markersize=5,
                            label="Secant Line" if not secant_line_label_shown else "")
                    secant_line_label_shown = True
                    ax.scatter(x2_i, f_lambda(x2_i), color='orange', s=50)
                    # Annotate the secant line at 60% along the segment, offset a bit
                    frac = 0.6
                    x_annot = x0_i + frac * (x1_i - x0_i)
                    y_annot = f_lambda(x0_i) + frac * (f_lambda(x1_i) - f_lambda(x0_i))
                    dx = x1_i - x0_i
                    dy = f_lambda(x1_i) - f_lambda(x0_i)
                    norm = (dx**2 + dy**2)**0.5
                    # Make the annotation further away and always use a pointer line
                    offset_multiplier = 4.0  # Further from the secant line
                    if norm != 0:
                        offset_x = -dy / norm * offset_multiplier
                        offset_y = dx / norm * offset_multiplier
                    else:
                        offset_x = offset_y = 0
                    ann = ax.annotate(
                        f"Secant {i+1}",
                        xy=(x_annot, y_annot),
                        xytext=(x_annot + offset_x, y_annot + offset_y),
                        color='#00bfff',
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="#00bfff", lw=1, alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color='#00bfff', lw=2.5, shrinkA=0, shrinkB=0),
                        ha='left', va='center'
                    )
                    draggable_annotations.append(DraggableAnnotation(ann))
            # Mark the root
            ax.scatter(root, f_lambda(root), color='lime', s=120, edgecolors='black', zorder=5,
                       label=f"Secant Root ≈ {root:.5f}" if not root_label_shown else "")
            root_label_shown = True
            ax.axvline(root, linestyle="dashed", color="lime", linewidth=1)
            # Stagger the annotation vertically to avoid overlap
            stagger = (idx % 5) * 18  # 18 pixels per root, change 5 to a higher number if many roots
            ann = ax.annotate(
                f"x ≈ {root:.5f}",
                (root, f_lambda(root)),
                textcoords="offset points",
                xytext=(0, 10 + stagger),
                ha='center',
                color='lime',
                fontsize=12,
                fontweight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="#23272e",
                    ec="lime",
                    lw=1,
                    alpha=0.8
                )
            )
            draggable_annotations.append(DraggableAnnotation(ann))
        # After plotting the function line
        x_annot = x_values[int(len(x_values) * 0.8)]  # 80% along the x-axis
        y_annot = y_values[int(len(y_values) * 0.8)]
        ann = ax.annotate(
            f"f(x) = {eq}",
            xy=(x_annot, y_annot),
            xytext=(x_annot + 1, y_annot + (max(y_values) - min(y_values)) * 0.1),  # Offset
            color='white',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="white", lw=1, alpha=0.8),
            arrowprops=dict(arrowstyle="->", color='white', lw=1),
            ha='left', va='center'
        )
        draggable_annotations.append(DraggableAnnotation(ann))
        # Grid, labels, and aesthetics
        ax.axhline(0, color='white', linewidth=1.2)
        ax.axvline(0, color='white', linewidth=1.2)
        ax.grid(True, linestyle='--', linewidth=0.7, color='gray')
        ax.set_xlabel("X-Axis", color='white', fontsize=12)
        ax.set_ylabel("Y-Axis", color='white', fontsize=12)
        ax.set_title(f"Secant Method: {eq}", color='white', fontsize=14)
        ax.legend(loc="upper left", fontsize=10)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        canvas.draw()
        # Update root label
        if roots:
            root_display.set_root_text("SECANT ROOTS: " + ", ".join([f"x ≈ {r:.6f}" for r in roots]))
        else:
            root_display.set_root_text("NO ROOTS FOUND")
        add_tip_annotation(ax, draggable_annotations)
    except Exception as e:
        messagebox.showerror("Error", f"Secant Method Error: {e}")

def run_newton_raphson_method():
    try:
        global draggable_annotations
        draggable_annotations.clear()
        eq = preprocess_equation(equation_entry.get())
        x_sym = sp.symbols('x')
        f_sympy = sp.sympify(eq, locals={'abs': sp.Abs, 'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        f_lambda = sp.lambdify(x_sym, f_sympy, 'numpy')
        df_sympy = sp.diff(f_sympy, x_sym)
        df_lambda = sp.lambdify(x_sym, df_sympy, 'numpy')

        a, b, step = get_limits()
        tol = 1e-12   # Tighter tolerance for higher accuracy
        max_iter = 200  # More iterations for accuracy
        step = 0.01  # Finer step for initial guesses
        roots = []
        all_iterations = []
        x_scan = np.arange(a, b, step)
        for x0 in x_scan:
            try:
                root, iterations = newton_raphson_method(f_lambda, df_lambda, x0, tol=tol, max_iter=max_iter)
                if root is not None and np.isfinite(root):
                    if a - step <= root <= b + step:
                        if not any(abs(root - r) < tol*10 for r in roots):
                            roots.append(root)
                            all_iterations.append(iterations)
            except Exception:
                continue
        # Plotting
        try:
            x_values = np.linspace(a, b, 1000)
            y_values = f_lambda(x_values)
        except Exception:
            messagebox.showerror("Error", "Function could not be evaluated for plotting. Please check your input.")
            return
        ax.clear()
        try:
            ax.plot(x_values, y_values, label=f"f(x) = {eq}", color='white', linewidth=2)
        except Exception:
            messagebox.showerror("Error", "Plotting failed. Please check your function.")
            return
        # Plot tangent lines and roots
        for idx, (root, iterations) in enumerate(zip(roots, all_iterations)):
            for i, (x0_i, x1_i) in enumerate(iterations):
                try:
                    y0 = f_lambda(x0_i)
                    slope = df_lambda(x0_i)
                    tangent_x = np.linspace(x0_i - 1, x0_i + 1, 10)
                    tangent_y = y0 + slope * (tangent_x - x0_i)
                    # Make tangent lines more visible with thicker lines and better color
                    ax.plot(tangent_x, tangent_y, color='#00ff00', linestyle='-', linewidth=2.5, 
                           label="Tangent Line" if i == 0 and idx == 0 else "")
                    # Add points to show the iteration points
                    ax.scatter(x0_i, y0, color='#ff00ff', s=100, zorder=5, 
                             label="Current Point" if i == 0 and idx == 0 else "")
                    ax.scatter(x1_i, f_lambda(x1_i), color='#00ffff', s=100, zorder=5,
                             label="Next Point" if i == 0 and idx == 0 else "")
                    ax.scatter(x1_i, f_lambda(x1_i), color='magenta', s=50)
                    # Annotate the tangent line at 60% along the segment, offset a bit
                    frac = 0.6
                    x_annot = tangent_x[0] + frac * (tangent_x[-1] - tangent_x[0])
                    y_annot = tangent_y[0] + frac * (tangent_y[-1] - tangent_y[0])
                    dx = tangent_x[-1] - tangent_x[0]
                    dy = tangent_y[-1] - tangent_y[0]
                    norm = (dx**2 + dy**2)**0.5
                    # Make the annotation further away and always use a pointer line
                    offset_multiplier = 4.0  # Further from the secant line
                    if norm != 0:
                        offset_x = -dy / norm * offset_multiplier
                        offset_y = dx / norm * offset_multiplier
                    else:
                        offset_x = offset_y = 0
                    ann = ax.annotate(
                        f"Tangent {i+1}",
                        xy=(x_annot, y_annot),
                        xytext=(x_annot + offset_x, y_annot + offset_y),
                        color='green',
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="green", lw=1, alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color='green', lw=2.5, shrinkA=0, shrinkB=0),
                        ha='left', va='center'
                    )
                    draggable_annotations.append(DraggableAnnotation(ann))
                except Exception:
                    continue
            try:
                ann = ax.annotate(
                    f"x ≈ {root:.5f}",
                    (root, f_lambda(root)),
                    textcoords="offset points",
                    xytext=(0, 10 + (idx % 5) * 18),
                    ha='center',
                    color='lime',
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="#23272e",
                        ec="lime",
                        lw=1,
                        alpha=0.8
                    ),
                    arrowprops=dict(arrowstyle="->", color='lime', lw=2.5, shrinkA=0, shrinkB=0)
                )
                draggable_annotations.append(DraggableAnnotation(ann))
            except Exception:
                continue
        # After plotting the function line
        try:
            x_annot = x_values[int(len(x_values) * 0.8)]  # 80% along the x-axis
            y_annot = y_values[int(len(y_values) * 0.8)]
            ann = ax.annotate(
                f"f(x) = {eq}",
                xy=(x_annot, y_annot),
                xytext=(x_annot + 1, y_annot + (max(y_values) - min(y_values)) * 0.1),  # Offset
                color='white',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="white", lw=1, alpha=0.8),
                arrowprops=dict(arrowstyle="->", color='white', lw=1),
                ha='left', va='center'
            )
            draggable_annotations.append(DraggableAnnotation(ann))
        except Exception:
            pass
        # Grid, labels, and aesthetics
        ax.axhline(0, color='white', linewidth=1.2)
        ax.axvline(0, color='white', linewidth=1.2)
        ax.grid(True, linestyle='--', linewidth=0.7, color='gray')
        ax.set_xlabel("X-Axis", color='white', fontsize=12)
        ax.set_ylabel("Y-Axis", color='white', fontsize=12)
        ax.set_title(f"Newton-Raphson Method: {eq}", color='white', fontsize=14)
        ax.legend(loc="upper left", fontsize=10)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        canvas.draw()
        # Update root label
        if roots:
            root_display.set_root_text("NEWTON ROOTS: " + ", ".join([f"x ≈ {r:.6f}" for r in roots]))
        else:
            root_display.set_root_text("NO ROOTS FOUND")
        add_tip_annotation(ax, draggable_annotations)
    except Exception as e:
        messagebox.showerror("Error", f"Newton-Raphson Error: {e}")

def run_regula_falsi_method():
    try:
        global draggable_annotations
        draggable_annotations.clear()
        eq = preprocess_equation(equation_entry.get())
        x_sym = sp.symbols('x')
        f_sympy = sp.sympify(eq, locals={'abs': sp.Abs, 'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        f_lambda = sp.lambdify(x_sym, f_sympy, 'numpy')

        a, b, step = get_limits()
        tol = 1e-7  # More accurate
        max_iter = 100  # More iterations for accuracy
        roots = []
        all_iterations = []
        intervals = []
        x_scan = np.arange(a, b, step)
        for i in range(len(x_scan) - 1):
            x0, x1 = x_scan[i], x_scan[i+1]
            try:
                f0, f1 = f_lambda(x0), f_lambda(x1)
                f0 = np.asarray(f0, dtype=float)
                f1 = np.asarray(f1, dtype=float)
                if np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                    continue
                if f0 * f1 < 0:
                    root, iterations = regula_falsi_method(f_lambda, x0, x1, tol=tol, max_iter=max_iter)
                    if root is not None:
                        if not any(abs(root - r) < tol*10 for r in roots):
                            roots.append(root)
                            all_iterations.append(iterations)
                            intervals.append((x0, x1))
            except Exception:
                continue
        # Plotting
        x_values = np.linspace(a, b, 1000)
        y_values = f_lambda(x_values)
        ax.clear()
        plt.style.use('ggplot')
        # Offsets for label spacing
        dx = (b - a) * 0.04
        dy = (max(y_values) - min(y_values)) * 0.08
        # Plot the function
        ax.plot(x_values, y_values, label=f"f(x) = {eq}", color='white', linewidth=2)
        ann_fx = ax.annotate("f(x)",
            xy=(x_values[-1], y_values[-1]),
            xytext=(x_values[-1] + dx, y_values[-1] + dy),
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="white", lw=1, alpha=0.8),
            arrowprops=dict(arrowstyle="->", color='white', lw=1),
            ha='left', va='center')
        draggable_annotations.append(DraggableAnnotation(ann_fx))
        # Plot Regula Falsi convergence for each root
        for idx, (root, iterations) in enumerate(zip(roots, all_iterations)):
            c_vals = [c for (a_i, b_i, c) in iterations]
            a_vals = [a_i for (a_i, b_i, c) in iterations]
            b_vals = [b_i for (a_i, b_i, c) in iterations]
            # Plot the movement of a and b as dashed lines
            ax.plot(a_vals, [f_lambda(a) for a in a_vals], 'g--', alpha=0.5, label='a (left bracket)' if idx == 0 else None)
            ax.plot(b_vals, [f_lambda(b) for b in b_vals], 'b--', alpha=0.5, label='b (right bracket)' if idx == 0 else None)
            # Plot the sequence of c-values as a line
            ax.plot(c_vals, [f_lambda(c) for c in c_vals], color='red', marker='o', markersize=5, linewidth=2, alpha=0.8, label='c (False Position)' if idx == 0 else None)
            # Legend-like labels for a, b, c with spacing
            if a_vals:
                ann_a = ax.annotate("a (left bracket)",
                    xy=(a_vals[-1], f_lambda(a_vals[-1])),
                    xytext=(a_vals[-1] - dx, f_lambda(a_vals[-1]) + dy),
                    color='green', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="green", lw=1, alpha=0.8),
                    arrowprops=dict(arrowstyle="->", color='green', lw=1),
                    ha='right', va='bottom')
                draggable_annotations.append(DraggableAnnotation(ann_a))
            if b_vals:
                ann_b = ax.annotate("b (right bracket)",
                    xy=(b_vals[-1], f_lambda(b_vals[-1])),
                    xytext=(b_vals[-1] + dx, f_lambda(b_vals[-1]) + dy),
                    color='blue', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="blue", lw=1, alpha=0.8),
                    arrowprops=dict(arrowstyle="->", color='blue', lw=1),
                    ha='left', va='bottom')
                draggable_annotations.append(DraggableAnnotation(ann_b))
            if c_vals:
                ann_c = ax.annotate("c (False Position)",
                    xy=(c_vals[-1], f_lambda(c_vals[-1])),
                    xytext=(c_vals[-1] + dx, f_lambda(c_vals[-1]) - dy),
                    color='red', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="red", lw=1, alpha=0.8),
                    arrowprops=dict(arrowstyle="->", color='red', lw=1),
                    ha='left', va='top')
                draggable_annotations.append(DraggableAnnotation(ann_c))
            # Mark the final root
            ax.scatter(root, f_lambda(root), color='lime', s=120, edgecolors='black', zorder=6, label=f"Regula Falsi Root ≈ {root:.7f}" if idx == 0 else None)
            # Stagger the annotation vertically to avoid overlap
            stagger = (idx % 5) * 18  # 18 pixels per root, change 5 to a higher number if many roots
            ann_root = ax.annotate(f"Regula Falsi Root ≈ {root:.7f}",
                xy=(root, f_lambda(root)),
                xytext=(root, f_lambda(root) + dy + stagger),
                color='lime', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="lime", lw=1, alpha=0.8),
                arrowprops=dict(arrowstyle="->", color='lime', lw=1),
                ha='center', va='bottom')
            draggable_annotations.append(DraggableAnnotation(ann_root))
            ax.axvline(root, linestyle="dashed", color="lime", linewidth=1.5)
            # Highlight the bracketing interval for the final root
            if len(a_vals) > 0 and len(b_vals) > 0:
                ax.axvspan(a_vals[-1], b_vals[-1], color='orange', alpha=0.2, label='Final Bracket' if idx == 0 else None)
                mid_x = (a_vals[-1] + b_vals[-1]) / 2
                mid_y = (f_lambda(a_vals[-1]) + f_lambda(b_vals[-1])) / 2
                ann_final = ax.annotate("Final Bracket",
                    xy=(mid_x, mid_y),
                    xytext=(mid_x, mid_y - dy),
                    color='orange', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="orange", lw=1, alpha=0.8),
                    arrowprops=dict(arrowstyle="->", color='orange', lw=1),
                    ha='center', va='top')
                draggable_annotations.append(DraggableAnnotation(ann_final))
        # After plotting the function line
        x_annot = x_values[int(len(x_values) * 0.8)]  # 80% along the x-axis
        y_annot = y_values[int(len(y_values) * 0.8)]
        ann_fx2 = ax.annotate(
            f"f(x) = {eq}",
            xy=(x_annot, y_annot),
            xytext=(x_annot + 1, y_annot + (max(y_values) - min(y_values)) * 0.1),  # Offset
            color='white',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="white", lw=1, alpha=0.8),
            arrowprops=dict(arrowstyle="->", color='white', lw=1),
            ha='left', va='center'
        )
        draggable_annotations.append(DraggableAnnotation(ann_fx2))
        # Grid, labels, and aesthetics
        ax.axhline(0, color='white', linewidth=1.2)
        ax.axvline(0, color='white', linewidth=1.2)
        ax.grid(True, linestyle='--', linewidth=0.7, color='gray')
        ax.set_xlabel("X-Axis", color='white', fontsize=12)
        ax.set_ylabel("Y-Axis", color='white', fontsize=12)
        ax.set_title(f"Regula Falsi Method: {eq}", color='white', fontsize=14)
        # Restore the legend box
        ax.legend(loc="upper left", fontsize=10)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        canvas.draw()
        # Update root label
        if roots:
            root_display.set_root_text("REGULA FALSI ROOTS: " + ", ".join([f"x ≈ {r:.7f}" for r in roots]))
        else:
            root_display.set_root_text("NO ROOTS FOUND")
        add_tip_annotation(ax, draggable_annotations)
    except Exception as e:
        messagebox.showerror("Error", f"Regula Falsi Error: {e}")

def show_instructions():
    instructions = (
        "How to Use the Iterative Bracketing Solver\n\n"
        "1. Choose Input Mode:\n"
        "   - Equation: Enter a function of x (e.g., x**3 - x - 2) in the input box.\n"
        "   - Data Points: Enter (x, y) pairs, one per line (e.g., 1,2), in the data points box.\n\n"
        "2. (Optional) Solve a System of Equations:\n"
        "   - Click the 'GRAPHICAL METHOD ACCURATE' button in the sidebar.\n"
        "   - Enter two equations in variables x1 and x2 (e.g., 3*x1 + 2*x2 = 18 and -x1 + 2*x2 = 2).\n"
        "   - The intersection point will be shown on the graph.\n\n"
        "3. Sidebar Actions:\n"
        "   - FIND ROOT: Finds roots by bracketing intervals and displays them.\n"
        "   - GENERATE TABLE: Shows a table of intervals and function values for bracketing.\n"
        "   - GENERATE GRAPH: Plots the function or interpolated data points.\n"
        "   - GRAPHICAL METHOD: Plots the function for visual root inspection.\n"
        "   - INCREMENTAL METHOD: Highlights intervals where roots are bracketed.\n"
        "   - SECANT METHOD, NEWTON-RAPHSON, REGULA FALSI: Run these numerical methods and visualize their steps and roots.\n\n"
        "4. Multiple Roots:\n"
        "   - The app will attempt to find and label all roots within the interval.\n\n"
        "5. Legends and Results:\n"
        "   - Roots and solution points are labeled on the graph.\n"
        "   - The table provides a step-by-step breakdown of intervals and function values.\n"
        "   - The result label below the graph displays the found roots or solutions.\n\n"
        "Tips:\n"
        "- For Data Points mode, at least two (x, y) pairs are required.\n"
        "- Use the RESET button to clear the graph, table, and results.\n"
        "- Use the INSTRUCTIONS button anytime for help.\n"
        "- Square root examples: sqrt(x), √x, √(x), x^(1/2), x^0.5\n"
    )
    # Create a custom Toplevel window
    instr_win = tk.Toplevel(root)
    instr_win.title("Instructions")
    instr_win.configure(bg="#f5f5f5")
    instr_win.geometry("600x900")
    instr_win.resizable(True, True)
    # Center the window on the screen
    instr_win.update_idletasks()
    w = instr_win.winfo_width()
    h = instr_win.winfo_height()
    ws = instr_win.winfo_screenwidth()
    hs = instr_win.winfo_screenheight()
    x = (ws // 2) - (w // 2)
    y = (hs // 2) - (h // 2)
    instr_win.geometry(f"600x900+{x}+{y}")
    # Add a bold label for the instructions
    instr_label = tk.Label(instr_win, text=instructions, font=("Segoe UI", 12, "bold"), justify="left", bg="#f5f5f5", anchor="nw", wraplength=580)
    instr_label.pack(padx=20, pady=20, fill="both", expand=True)
    # Add an OK button to close the window
    ok_btn = tk.Button(instr_win, text="OK", font=("Segoe UI", 12, "bold"), command=instr_win.destroy, bg="#ff9800", fg="#fff", activebackground="#ffa726", activeforeground="#fff", relief="solid", borderwidth=0, padx=20, pady=8)
    ok_btn.pack(pady=(0, 20))
    instr_win.transient(root)
    instr_win.grab_set()
    instr_win.focus_set()

def reset_results():
    # Clear the graph
    ax.clear()
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    canvas.draw()
    # Clear the table
    for i in table.get_children():
        table.delete(i)
    # Reset the root label
    root_display.set_root_text("ROOT WILL BE DISPLAYED HERE")

# --- Utility functions for showing/unhiding widgets ---
def show(widget, **pack_options):
    """Show a widget using pack with optional pack options (default: pady=20)."""
    if not pack_options:
        pack_options = {'pady': 20}
    widget.pack(**pack_options)

def unhide(widget, **pack_options):
    """Alias for show, for semantic clarity."""
    show(widget, **pack_options)

# --- Helper for button animation ---
def animate_button_color(widget, from_color, to_color, duration=300, steps=18):
    # Cancel any previous color animation
    if hasattr(widget, '_color_anim_job') and widget._color_anim_job:
        widget.after_cancel(widget._color_anim_job)
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb
    from_rgb = hex_to_rgb(from_color)
    to_rgb = hex_to_rgb(to_color)
    def interpolate(a, b, t):
        t = t * t * (3 - 2 * t)
        return int(a + (b - a) * t)
    def step(i):
        t = i / steps
        rgb = tuple(interpolate(f, t_, t) for f, t_ in zip(from_rgb, to_rgb))
        widget.configure(background=rgb_to_hex(rgb))
        if i < steps:
            widget._color_anim_job = widget.after(duration // steps, step, i + 1)
        else:
            widget._color_anim_job = None
    widget._color_anim_job = widget.after(0, step, 0)


def animate_button_size(btn, from_size, to_size, from_pad, to_pad, from_border, to_border, duration=300, steps=18):
    # Cancel any previous size animation
    if hasattr(btn, '_size_anim_job') and btn._size_anim_job:
        btn.after_cancel(btn._size_anim_job)
    def interpolate(a, b, t):
        t = t * t * (3 - 2 * t)
        return int(a + (b - a) * t)
    def step(i):
        t = i / steps
        size = interpolate(from_size, to_size, t)
        padx = interpolate(from_pad[0], to_pad[0], t)
        pady = interpolate(from_pad[1], to_pad[1], t)
        border = interpolate(from_border, to_border, t)
        btn.config(font=("Segoe UI", size, "bold"), padx=padx, pady=pady, borderwidth=border)
        if i < steps:
            btn._size_anim_job = btn.after(duration // steps, step, i + 1)
        else:
            btn._size_anim_job = None
    btn._size_anim_job = btn.after(0, step, 0)


def add_animated_effects(btn, base_color, hover_color, active_color):
    orig_font = btn.cget('font')
    try:
        orig_size = int(str(orig_font).split()[1])
    except Exception:
        orig_size = 16
    orig_padx = btn.cget('padx') if 'padx' in btn.keys() else 10
    orig_pady = btn.cget('pady') if 'pady' in btn.keys() else 8
    try:
        orig_padx = int(orig_padx)
    except Exception:
        orig_padx = 10
    try:
        orig_pady = int(orig_pady)
    except Exception:
        orig_pady = 8
    orig_border = btn.cget('borderwidth') if 'borderwidth' in btn.keys() else 0
    try:
        orig_border = int(orig_border)
    except Exception:
        orig_border = 0
    grow_size = orig_size + 2
    grow_pad = (orig_padx + 12, orig_pady + 6)
    grow_border = orig_border + 4
    press_size = grow_size
    press_pad = (grow_pad[0], grow_pad[1])
    press_border = grow_border
    orig_pad = (orig_padx, orig_pady)

    # Store original width for reset
    orig_width = btn.winfo_width()
    # Helper to check if this is a sidebar method button (by font size and parent bg)
    def is_sidebar_method_button():
        return btn.cget('font') == ("Segoe UI", 13, "bold") or btn.cget('font') == 'Segoe UI 13 bold'

    def on_enter(e):
        animate_button_color(btn, base_color, hover_color, duration=300, steps=18)
        animate_button_size(btn, orig_size, grow_size, orig_pad, grow_pad, orig_border, grow_border, duration=300, steps=18)
        if is_sidebar_method_button():
            btn.update_idletasks()
            text_width = btn.winfo_reqwidth()
            current_width = btn.winfo_width()
            # Add a little extra for padding
            if text_width > current_width:
                btn.config(width=text_width)
    def on_leave(e):
        animate_button_color(btn, hover_color, base_color, duration=300, steps=18)
        animate_button_size(btn, grow_size, orig_size, grow_pad, orig_pad, grow_border, orig_border, duration=300, steps=18)
        if is_sidebar_method_button():
            btn.config(width=orig_width)
    def on_press(e):
        animate_button_color(btn, hover_color, active_color, duration=180, steps=10)
        animate_button_size(btn, grow_size, press_size, grow_pad, press_pad, grow_border, press_border, duration=180, steps=10)
    def on_release(e):
        animate_button_color(btn, active_color, hover_color, duration=180, steps=10)
        animate_button_size(btn, press_size, grow_size, press_pad, grow_pad, press_border, grow_border, duration=180, steps=10)
    btn.bind('<Enter>', on_enter)
    btn.bind('<Leave>', on_leave)
    btn.bind('<ButtonPress-1>', on_press)
    btn.bind('<ButtonRelease-1>', on_release)

def plot_function_accurately(f_lambda, a=-10, b=10, root=None, root_label=None):
    global draggable_annotations
    draggable_annotations.clear()
    x_values = np.linspace(a, b, 2000)
    y_values = f_lambda(x_values)
    ax.clear()
    ax.plot(x_values, y_values, label="f(x)", color='white', linewidth=2)
    ax.axhline(0, color='lime', linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(0, color='white', linewidth=1.2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray')
    ax.set_xlabel("X-Axis", color='white', fontsize=12)
    ax.set_ylabel("Y-Axis", color='white', fontsize=12)
    ax.set_title("Numerical Methods Graph: f(x)", color='white', fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.tick_params(axis='both', colors='white', labelsize=12)
    ax.set_xlim(a, b)
    # Optionally set y-limits for better focus
    # ax.set_ylim(min(y_values), max(y_values))
    if root is not None:
        ax.scatter(root, f_lambda(root), color='lime', s=100, edgecolors='black')
        ax.axvline(root, linestyle="dashed", color="lime", linewidth=1)
        ann_root = ax.annotate(
            root_label or f"x ≈ {root:.5f}",
            (root, f_lambda(root)),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            color='lime',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="lime", lw=1, alpha=0.8)
        )
        draggable_annotations.append(DraggableAnnotation(ann_root))
    # After plotting the function line
    x_annot = x_values[int(len(x_values) * 0.8)]  # 80% along the x-axis
    y_annot = y_values[int(len(y_values) * 0.8)]
    ann_fx = ax.annotate(
        f"f(x)",
        xy=(x_annot, y_annot),
        xytext=(x_annot + 1, y_annot + (max(y_values) - min(y_values)) * 0.1),  # Offset
        color='white',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="#23272e", ec="white", lw=1, alpha=0.8),
        arrowprops=dict(arrowstyle="->", color='white', lw=1),
        ha='left', va='center'
    )
    draggable_annotations.append(DraggableAnnotation(ann_fx))
    canvas.draw()
    add_tip_annotation(ax, draggable_annotations)

# --- Advanced Root Display Widget ---
class RootDisplay(tk.Frame):
    def __init__(self, master, width=500, height=110, **kwargs):
        super().__init__(master, bg='#181818', **kwargs)
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self, width=width, height=height, bg='#181818', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.rounded_rect = None
        self.shadow = None
        self.icon_id = None
        self.text_id = None
        self._draw_card()
        self.set_root_text("ROOT WILL BE DISPLAYED HERE")

    def _draw_card(self):
        # Draw shadow
        self.canvas.create_oval(18, 18, self.width-8, self.height-8, fill="#111", outline="", width=0)
        # Draw rounded rectangle (card)
        self._draw_rounded_rect(8, 8, self.width-16, self.height-16, 28, fill="#23272e", outline="#00ff99", width=3)

    def _draw_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        # Draw a rounded rectangle on the canvas
        points = [
            x1+r, y1,
            x2-r, y1,
            x2, y1,
            x2, y1+r,
            x2, y2-r,
            x2, y2,
            x2-r, y2,
            x1+r, y2,
            x1, y2,
            x1, y2-r,
            x1, y1+r,
            x1, y1
        ]
        self.canvas.create_polygon(points, smooth=True, **kwargs)

    def set_root_text(self, text, fg="#00ff99"):
        self.canvas.delete("icon")
        self.canvas.delete("text")
        # Larger font sizes
        icon_font = ("Segoe UI", 28, "bold")
        text_font = ("Segoe UI", 20, "bold")
        # Measure text and icon for alignment
        temp_id = self.canvas.create_text(0, 0, text="√", font=icon_font, anchor="nw")
        icon_bbox = self.canvas.bbox(temp_id)
        icon_width = icon_bbox[2] - icon_bbox[0] if icon_bbox else 36
        icon_height = icon_bbox[3] - icon_bbox[1] if icon_bbox else 28
        self.canvas.delete(temp_id)
        temp_id = self.canvas.create_text(0, 0, text=text, font=text_font, anchor="nw")
        text_bbox = self.canvas.bbox(temp_id)
        text_width = text_bbox[2] - text_bbox[0] if text_bbox else 180
        text_height = text_bbox[3] - text_bbox[1] if text_bbox else 20
        self.canvas.delete(temp_id)
        total_width = icon_width + 16 + text_width
        min_width = 500
        new_width = max(min_width, total_width + 32)
        if new_width != self.width:
            self.width = new_width
            self.canvas.config(width=self.width)
            self.config(width=self.width)
            self.canvas.delete("all")
            self._draw_card()
        x_start = (self.width - total_width) // 2
        # Align icon and text on the same baseline
        y_center = (self.height - text_height) // 2
        icon_y = y_center + (text_height - icon_height) // 2
        self.icon_id = self.canvas.create_text(x_start, icon_y, text="√", font=icon_font, fill=fg, tags="icon", anchor="nw")
        self.text_id = self.canvas.create_text(x_start + icon_width + 16, y_center, text=text, font=text_font, fill="#fff", tags="text", anchor="nw")

def update_secant_table(iterations):
    # Clear table
    for i in table.get_children():
        table.delete(i)
    # Insert new rows
    for idx, (x0, x1, x2) in enumerate(iterations, 1):
        f0 = f_lambda(x0)
        f1 = f_lambda(x1)
        f2 = f_lambda(x2)
        error = abs(x2 - x1)
        table.insert("", "end", values=(
            idx, f"{x0:.6f}", f"{x1:.6f}", f"{x2:.6f}",
            f"{f0:.6f}", f"{f1:.6f}", f"{f2:.6f}", f"{error:.2e}"
        ))

def update_newton_table(iterations):
    for i in table.get_children():
        table.delete(i)
    for idx, (x0, x1) in enumerate(iterations, 1):
        f0 = f_lambda(x0)
        df0 = df_lambda(x0)
        error = abs(x1 - x0)
        table.insert("", "end", values=(
            idx, f"{x0:.6f}", f"{x1:.6f}", f"{f0:.6f}", f"{df0:.6f}", f"{error:.2e}"
        ))

def add_tip_annotation(ax, draggable_annotations, x=None, y=None, text=None):
    if text is None:
        text = "TIP: Drag this label to reposition. You can drag any label on the graph!"
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Place at the bottom center
    if x is None:
        x = xlim[0] + (xlim[1] - xlim[0]) * 0.5  # Center horizontally
    if y is None:
        y = ylim[0] + (ylim[1] - ylim[0]) * 0.08  # Near bottom
    ann = ax.annotate(
        text,
        xy=(x, y),
        xytext=(x, y),
        color='#ffd600',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#23272e", ec="#ffd600", lw=2, alpha=0.95),
        ha='center',  # Center align text
        va='bottom',
        zorder=1000  # Ensure TIP is always on top
    )
    draggable_annotations.append(DraggableAnnotation(ann))
    # Force redraw to ensure TIP is visible
    ax.figure.canvas.draw_idle()

root = ThemedTk(theme="black")
root.title("Iterative Bracketing Solver")
root.geometry("1600x1000")
root.configure(bg='#181818')
# root.iconbitmap('')  # Add your icon path if available

# --- Main Layout ---
main_frame = tk.Frame(root, bg='#181818')
main_frame.pack(fill='both', expand=True)

# --- Sidebar ---
sidebar_frame = tk.Frame(main_frame, bg='#23272e', width=300)
sidebar_frame.pack(side='left', fill='y', anchor='center')

# --- Content Frame (everything else goes here) ---
content_frame = tk.Frame(main_frame, bg='#181818')
content_frame.pack(side='left', fill='both', expand=True)

# --- Style ---
style = ttk.Style()
style.theme_use('clam')
# Custom styles for orange and red buttons
style.configure('Orange.TButton', font=("Segoe UI", 16, "bold"), padding=10, background='#ff9800', foreground='#fff', borderwidth=0)
style.map('Orange.TButton', background=[('active', '#ffa726')])
style.configure('Red.TButton', font=("Segoe UI", 16, "bold"), padding=10, background='#e53935', foreground='#fff', borderwidth=0)
style.map('Red.TButton', background=[('active', '#ef5350')])
style.configure('TButton', font=("Segoe UI", 16, "bold"), padding=10, background='#222', foreground='#fff', borderwidth=0)
style.map('TButton', background=[('active', '#444')])
style.configure('TLabel', font=("Segoe UI", 16), background='#181818', foreground='#fff')
style.configure('Treeview', font=("Segoe UI", 14), rowheight=32, background='#222', fieldbackground='#222', foreground='#fff')
style.configure('Treeview.Heading', font=("Segoe UI", 16, "bold"), background='#333', foreground='#fff')
style.map('Treeview', background=[('selected', '#444')])

# --- Padding and Frames ---
top_frame = tk.Frame(content_frame, bg="#181818")
top_frame.pack(pady=20)
input_frame = tk.Frame(content_frame, bg="#181818")
input_frame.pack(pady=20)
button_frame = tk.Frame(content_frame, bg="#181818")
button_frame.pack(pady=20)

root_display = RootDisplay(content_frame, width=520, height=80)
root_display.pack(pady=30)

graph_table_frame = tk.Frame(content_frame, bg="#181818")
graph_table_frame.pack(pady=20, fill="both", expand=True)

# --- Redesigned Input Card ---
input_card_frame = tk.Frame(top_frame, bg='#23272e', bd=0, highlightthickness=0)
input_card_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
input_card_frame.grid_columnconfigure(1, weight=1)

# Math icon (using Unicode or text)
icon_label = tk.Label(input_card_frame, text='ƒ(x)', font=("Segoe UI", 32, "bold"), fg="#00ff99", bg="#23272e")
icon_label.grid(row=0, column=0, rowspan=2, padx=(24, 18), pady=18, sticky='n')

# Main label
main_label = tk.Label(input_card_frame, text="Enter a function of x", font=("Segoe UI", 20, "bold"), fg="#fff", bg="#23272e", anchor='w')
main_label.grid(row=0, column=1, sticky='w', pady=(18, 0))

# Hint label
hint_label = tk.Label(input_card_frame, text="e.g. x**3 - x - 2, sqrt(x), √x, x^(1/2), x^0.5", font=("Segoe UI", 14), fg="#aaa", bg="#23272e", anchor='w')
hint_label.grid(row=1, column=1, sticky='w', pady=(0, 10))

# Styled Entry
# Add a variable for text size
EQUATION_ENTRY_FONT_SIZE = 25  # Adjust as desired
style.configure('Card.TEntry', font=("Segoe UI", EQUATION_ENTRY_FONT_SIZE), fieldbackground='#181818', foreground='#00ff99', borderwidth=0, relief='flat')
equation_entry = ttk.Entry(input_card_frame, width=30, style='Card.TEntry')
equation_entry.grid(row=2, column=0, columnspan=2, padx=24, pady=(0, 28), sticky='ew', ipady=12)
equation_entry.insert(0, " x**3 - x - 2")
# Ensure the inserted text uses the correct font size
try:
    equation_entry.configure(font=("Segoe UI", EQUATION_ENTRY_FONT_SIZE))
except Exception:
    pass  # ttk.Entry may not support direct font config on all platforms

# --- Lower/Upper Limit and Step Size Inputs ---
limit_frame = tk.Frame(input_card_frame, bg="#23272e")
limit_frame.grid(row=3, column=0, columnspan=2, padx=24, pady=(0, 18), sticky="ew")

BIG_INPUT_FONT = ("Segoe UI", 20, "bold")
ENTRY_WIDTH = 12

lower_label = tk.Label(limit_frame, text="Lower Limit:", font=BIG_INPUT_FONT, fg="#fff", bg="#23272e")
lower_label.grid(row=0, column=0, padx=(0, 16), sticky='e')
lower_entry = ttk.Entry(limit_frame, width=ENTRY_WIDTH, style='Card.TEntry', font=BIG_INPUT_FONT)
lower_entry.grid(row=0, column=1, padx=(0, 32), ipady=8, ipadx=2)
lower_entry.delete(0, tk.END)
lower_entry.insert(0, "-10")

upper_label = tk.Label(limit_frame, text="Upper Limit:", font=BIG_INPUT_FONT, fg="#fff", bg="#23272e")
upper_label.grid(row=0, column=2, padx=(0, 16), sticky='e')
upper_entry = ttk.Entry(limit_frame, width=ENTRY_WIDTH, style='Card.TEntry', font=BIG_INPUT_FONT)
upper_entry.grid(row=0, column=3, padx=(0, 32), ipady=8, ipadx=2)
upper_entry.delete(0, tk.END)
upper_entry.insert(0, "10")

step_label = tk.Label(limit_frame, text="Step Size:", font=BIG_INPUT_FONT, fg="#fff", bg="#23272e")
step_label.grid(row=0, column=4, padx=(0, 16), sticky='e')
step_entry = ttk.Entry(limit_frame, width=ENTRY_WIDTH, style='Card.TEntry', font=BIG_INPUT_FONT)
step_entry.grid(row=0, column=5, ipady=8, ipadx=2)
step_entry.delete(0, tk.END)
step_entry.insert(0, "0.1")

# Focus effect for Entry (glow border)
def on_entry_focus_in(event):
    input_card_frame.config(highlightbackground="#ffa726", highlightcolor="#ffa726", highlightthickness=3)
def on_entry_focus_out(event):
    input_card_frame.config(highlightbackground="#00ff99", highlightcolor="#00ff99", highlightthickness=3)
equation_entry.bind('<FocusIn>', on_entry_focus_in)
equation_entry.bind('<FocusOut>', on_entry_focus_out)
input_card_frame.config(highlightbackground="#00ff99", highlightcolor="#00ff99", highlightthickness=3)

# --- Sidebar ---
# Helper to create a fixed-size frame for each button
def create_button_frame(parent, width=300, height=80):
    f = tk.Frame(parent, width=width, height=height, bg=parent['bg'])
    f.pack_propagate(False)
    return f

actions_frame = tk.Frame(sidebar_frame, bg='#23272e')

def toggle_actions():
    if actions_frame.winfo_ismapped():
        actions_frame.pack_forget()
    else:
        actions_frame.pack(padx=10, pady=8, fill='x', expand=True)

instr_frame = create_button_frame(sidebar_frame, width=300, height=80)
instr_frame.pack(padx=10, pady=(24,12), anchor='center')
instr_btn = tk.Button(instr_frame, text="INSTRUCTIONS", command=show_instructions, bg='#ff9800', fg='#fff', font=("Segoe UI", 20, "bold"), relief='solid', borderwidth=0, activebackground='#ffa726', activeforeground='#fff', padx=24, pady=12)
instr_btn.pack(expand=True)
add_animated_effects(instr_btn, '#ff9800', '#ffa726', '#ffd180')

reset_frame = create_button_frame(sidebar_frame, width=300, height=80)
reset_frame.pack(padx=10, pady=12, anchor='center')
reset_btn = tk.Button(reset_frame, text="RESET", command=reset_results, bg='#e53935', fg='#fff', font=("Segoe UI", 20, "bold"), relief='solid', borderwidth=0, activebackground='#ef5350', activeforeground='#fff', padx=24, pady=12)
reset_btn.pack(expand=True)
add_animated_effects(reset_btn, '#e53935', '#ef5350', '#ff867c')

actions_toggle_frame = create_button_frame(sidebar_frame, width=300, height=80)
actions_toggle_frame.pack(padx=10, pady=12, anchor='center')
actions_toggle_btn = tk.Button(actions_toggle_frame, text="METHODS", command=toggle_actions, bg='#222222', fg='#fff', font=("Segoe UI", 20, "bold"), relief='solid', borderwidth=0, activebackground='#444444', activeforeground='#fff', padx=24, pady=12)
actions_toggle_btn.pack(expand=True)
add_animated_effects(actions_toggle_btn, '#222222', '#444444', '#888888')

sidebar_buttons = [
    ("FIND ROOT", find_root),
    ("GENERATE TABLE", generate_table),
    ("GENERATE GRAPH", generate_graph),
    ("GRAPHICAL METHOD", run_graphical_method),
    ("INCREMENTAL METHOD", run_incremental_method),
    ("BISECTION METHOD", run_bisection_method),
    ("SECANT METHOD", run_secant_method),
    ("NEWTON-RAPHSON METHOD", run_newton_raphson_method),
    ("REGULA FALSI METHOD", run_regula_falsi_method),
]
for text, cmd in sidebar_buttons:
    btn_frame = create_button_frame(actions_frame)
    btn_frame.pack(padx=10, pady=8)
    btn = tk.Button(
        btn_frame,
        text=text,
        command=cmd,
        bg='#222222',
        fg='#fff',
        font=("Segoe UI", 13, "bold"),
        relief='solid',
        borderwidth=0,
        activebackground='#444444',
        activeforeground='#fff',
        wraplength=260,  # Wrap text to fit button width
        justify='center' # Center the text
    )
    btn.pack(expand=True, fill='both')
    add_animated_effects(btn, '#222222', '#444444', '#888888')

# --- Graph ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_facecolor("#181818")
fig.patch.set_facecolor("#181818")
canvas = FigureCanvasTkAgg(fig, master=graph_table_frame)
canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=20, pady=20)

# --- Table ---
columns = ("Iteration", "x₁", "Δx", "xᵤ", "f(x₁)", "f(xᵤ)", "f(x₁)*f(xᵤ)", "Remark")
table_frame = tk.Frame(graph_table_frame, bg="#181818")
table_frame.pack(side="right", fill="both", expand=False, padx=20, pady=20)

scrollbar = ttk.Scrollbar(table_frame, orient="vertical")
table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10, yscrollcommand=scrollbar.set, style="Treeview")
scrollbar.config(command=table.yview)
scrollbar.pack(side="right", fill="y")

column_widths = [80, 100, 80, 100, 120, 120, 150, 350]
for col, width in zip(columns, column_widths):
    table.heading(col, text=col, anchor="center")
    table.column(col, width=width, anchor="center")

table.pack(side="right", fill="both", expand=True)

# Configure table tags after table creation
table.tag_configure('evenrow', background='#23272e')
table.tag_configure('oddrow', background='#181818')
table.tag_configure("remark", background="#e53935", foreground="#fff")

def highlight_remarks():
    for item in table.get_children():
        values = table.item(item, "values")
        if "⚠️" in values[-1]:
            table.item(item, tags=("remark",))
        else:
            idx = table.index(item)
            tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
            table.item(item, tags=(tag,))

def set_table_stripes():
    for i, item in enumerate(table.get_children()):
        table.item(item, tags=('evenrow' if i%2==0 else 'oddrow',))

table.bind('<<TreeviewSelect>>', set_table_stripes)

# --- Mode selection ---
mode_var = tk.StringVar(value='equation')
mode_frame = tk.Frame(content_frame, bg='#181818')
mode_frame.pack(pady=5)
tk.Label(mode_frame, text="Input Mode:", bg='#181818', fg='#fff', font=("Segoe UI", 14, "bold")).pack(side='left', padx=5)
tk.Radiobutton(mode_frame, text="Equation", variable=mode_var, value='equation', bg='#181818', fg='#fff', selectcolor='#23272e', font=("Segoe UI", 14)).pack(side='left', padx=5)
tk.Radiobutton(mode_frame, text="Data Points", variable=mode_var, value='datapoints', bg='#181818', fg='#fff', selectcolor='#23272e', font=("Segoe UI", 14)).pack(side='left', padx=5)

# --- Data points input UI ---
data_points_frame = tk.Frame(content_frame, bg='#181818')
data_points_label = tk.Label(data_points_frame, text="Enter data points (x,y) one per line, e.g. 1,2", bg='#181818', fg='#fff', font=("Segoe UI", 14))
data_points_label.pack()
data_points_text = tk.Text(data_points_frame, width=30, height=6, font=("Segoe UI", 14), bg='#fff', fg='#181818')
data_points_text.pack()

def update_input_mode(*args):
    if mode_var.get() == 'equation':
        data_points_frame.pack_forget()
    else:
        data_points_frame.pack(pady=20)

mode_var.trace_add('write', update_input_mode)

# --- System of Equations Input ---
system_frame = tk.Frame(content_frame, bg='#181818')
tk.Label(system_frame, text="Enter Equation 1 (in x1, x2):", bg='#181818', fg='#fff', font=("Segoe UI", 14)).pack(pady=2)
system_eq1_entry = ttk.Entry(system_frame, width=30, font=("Segoe UI", 14))
system_eq1_entry.pack(pady=2)
system_eq1_entry.insert(0, "3*x1 + 2*x2 = 18")
tk.Label(system_frame, text="Enter Equation 2 (in x1, x2):", bg='#181818', fg='#fff', font=("Segoe UI", 14)).pack(pady=2)
system_eq2_entry = ttk.Entry(system_frame, width=30, font=("Segoe UI", 14))
system_eq2_entry.pack(pady=2)
system_eq2_entry.insert(0, "-x1 + 2*x2 = 2")

# --- System Plotting Function ---
def plot_system_of_equations():
    try:
        eq1 = system_eq1_entry.get().replace('^', '**')
        eq2 = system_eq2_entry.get().replace('^', '**')
        x1, x2 = sp.symbols('x1 x2')
        # Parse equations
        if '=' in eq1:
            left1, right1 = eq1.split('=', 1)
            eq1 = f"({left1})-({right1})"
        if '=' in eq2:
            left2, right2 = eq2.split('=', 1)
            eq2 = f"({left2})-({right2})"
        expr1 = sp.sympify(eq1, locals={'abs': sp.Abs, 'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        expr2 = sp.sympify(eq2, locals={'abs': sp.Abs, 'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        # Solve system
        sol = sp.solve([expr1, expr2], (x1, x2), dict=True)
        if not sol:
            raise ValueError("No intersection found.")
        sol = sol[0]
        x1_sol, x2_sol = float(sol[x1]), float(sol[x2])
        # Express x2 in terms of x1 for both equations
        x1_vals = np.linspace(0, 7, 400)
        f1 = sp.solve(expr1, x2)
        f2 = sp.solve(expr2, x2)
        if not f1 or not f2:
            raise ValueError("Could not solve for x2.")
        f1_func = sp.lambdify(x1, f1[0], 'numpy')
        f2_func = sp.lambdify(x1, f2[0], 'numpy')
        x2_1 = f1_func(x1_vals)
        x2_2 = f2_func(x1_vals)
        # Plot
        ax.clear()
        ax.plot(x1_vals, x2_1, label=system_eq1_entry.get(), linewidth=3)
        ax.plot(x1_vals, x2_2, label=system_eq2_entry.get(), linewidth=3)
        ax.plot(x1_sol, x2_sol, 'ko', markersize=10)
        ax.text(x1_sol+0.2, x2_sol, f'Solution: x1={x1_sol:.2f}, x2={x2_sol:.2f}', fontsize=14, color='white', bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))
        ax.set_xlabel('$x_1$', color='white', fontsize=14)
        ax.set_ylabel('$x_2$', color='white', fontsize=14)
        ax.set_xlim(0,7)
        ax.set_ylim(0,9)
        ax.legend()
        leg = ax.legend()
        for text in leg.get_texts():
            text.set_color('black')
        ax.tick_params(axis='both', colors='white')
        ax.grid(True)
        ax.set_title('GRAPHICAL METHOD 2', color='white', fontsize=14)
        canvas.draw()
        root_display.set_root_text(f"Solution: x1 = {x1_sol:.4f}, x2 = {x2_sol:.4f}")
        add_tip_annotation(ax, draggable_annotations)
    except Exception as e:
        messagebox.showerror("Error", f"System Plot Error: {e}")

# --- Add Sidebar Button ---
def show_system_frame():
    show(system_frame, pady=20)

system_btn_frame = create_button_frame(actions_frame, width=300, height=100)
system_btn_frame.pack(padx=10, pady=8)
system_btn = tk.Button(
    system_btn_frame,
    text="GRAPHICAL METHOD ACCURATE",
    command=lambda: [show_system_frame(), plot_system_of_equations()],
    bg='#222222',
    fg='#fff',
    font=("Segoe UI", 13, "bold"),
    relief='solid',
    borderwidth=0,
    activebackground='#444444',
    activeforeground='#fff',
    wraplength=260,
    justify='center',
    padx=10,
    pady=10
)
system_btn.pack(expand=True)
add_animated_effects(system_btn, '#222222', '#444444', '#888888')

draggable_annotations = []

root.mainloop()