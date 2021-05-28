# Dual-Objective-Resdistricting

In this repository is the paper "Dual Objective Optimization in Political Redistricting", along with the supplemental code.

In the folder "newpopulation", there is a document explaining a new proposed method for the U and L constraints in the experiment, that is necessary to find more points in the experiment than the ones that are currently offered. It explains why the change to the U and L constraints are necessary for this research, and some of the details about what remains unknown about this new method (i.e., time complexities, complications, etc.). There is also a note explaining a current issue with the method, including results and maps that show the problem, and some potential causes.

I would like this code to be reviewed and fixed, as this method may be what I want to use in my initial experiments in finding the pareto-optimal solution for optimization in minimizing cut edges and minimizing deviation in all feasible states previously found. I do not know what the bug is, or how it could potentially be fixed, but further review by Dr. Buchanan and/or external parties would be very helpful in reviewing the code.

Review over whether the logic used in imposing this new constraint is valid or not, and the practicality of it would also be appreciated. I have little background in time-complexity analysis. Granted the results of at least the first experiment for IA yield a stronger optimization for cut edges and fall within the population deviation we desire, I think this program should work to a degree, if we can find how to implement it.
