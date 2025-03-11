Firstly for each test we define a set of default parameters for each test to run using, then we will modify one of these parameter and observe the difference.

```
population_size = 20
time_budget = 2.0
mutation_rate = 0.2
```

Each test was performed with 100 repetitions as per the instructions. The chosen dataset from the MAXSAT MES17 complete unweighted benchmarks was `TWComp_1c75_N69.wcnf`, the test where ran on the entirety of this file for increased accuracy over a subset of the file.

The mutation rates that where tested are: 0.1, 0.2, 0.3, 0.4, 0.5. You can observe this graph here:

![images/mutation_rate_paremeter_impact.png](Mutation Rate Parameter Impact)

From this you can see when the mutation rate was set to 0.1, on average the most algorithms where satisfied. -- whats the evolutionary algorithm doing