from typing import Callable, Tuple


def newton_raphson(
    f: Callable,
    df: Callable,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[int, float, float, bool]:
    """
    Performs the Newton-Raphson method to find the root of a function.

    Parameters
    ----------
    f : function
        The function whose root needs to be found.
    df : function
        The derivative of the function.
    x0 : float
        Initial guess.
    tol: float (default: 1e-6)
        Tolerance for convergence.
    max_iter: int (default 100)
        Maximum number of iterations.

    Returns
    -------
    iteration: int
        Number of iterations.
    x: float
        Final estimated value
    residual: float
        Final residual
    converged: bool
        Boolean indicating if solution is converged.
    """
    x = x0
    iteration = 0

    while abs(f(x)) > tol and iteration < max_iter:
        residual = f(x)
        x = x - residual / df(x)
        iteration += 1

    residual = f(x)
    converged = not abs(residual) > tol

    return iteration, x, residual, converged
