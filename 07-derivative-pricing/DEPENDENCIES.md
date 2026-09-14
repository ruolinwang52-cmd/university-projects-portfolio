# MATLAB dependencies

All project-specific pricing and simulation functions are defined locally at the end of `lab1.m` and `lab2.m`.

The scripts use standard MATLAB statistics functions including `binopdf` and `normcdf`. One optional comparison in `lab2.m` checks for the course-supplied compiled helper `opt_price.p`; when it is unavailable, that comparison is skipped and the rest of the script remains self-contained.
