# Alfven-Wave-Modes

The goal of this project was to demonstrate that a new cylindrical coordinate approach is justified
for the simulation of Alfven wave modes in a tokamak plasma. In order to approach the prob-
lem, the linear force operator was shown to follow from the equations of ideal MHD. It was
then shown that the linear force operator could be solved via a finite difference approximation
in which the expression is considered as a sequence of computational matrix operations. By
calling a library into the Python programming language, the ‘discretised’ expression could then
be solved via the use of the Arnoldi algorithm, which computes the solution of sparse eigen
problems.
With a working eigenmode code in place, three separate cases were studied to investigate
whether this new approach was suitable: the cases of a homogeneous plasma, an inhomogeneous
plasma and an inhomogeneous plasma with a supplied current.

This project is primarily a PoC to be extended to C++/Fortran for more complex boundary configurations.
