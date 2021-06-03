# Dual-Objective-Resdistricting

In this repository is the paper "Dual Objective Optimization in Political Redistricting", along with the supplemental code.

Two methods are currently under investigation as a way to measure the population deviation, known as the "unbalanced window", and "balanced window" methods.

The balanced window method creates a window in which population is allowed to deviate from an ideal population by the same percentage on either side. For example, at a 1.00% deviation, the allowed population of a district would be within +-0.50% of the ideal population. This shrinks as the program reiterates.

The unbalanced window defines a deviation as being a certain percentage of the ideal population, and then allows the district populations to fall in any range, so long as the difference in the largest and smallest districts returns a number lower than the allowed deviation.

A folder has been created for each method. When designing a configuration, we will include a key that determines unbalanced or balanced, and make changes to the programs necessary to accomodate to either at the user's preference.
