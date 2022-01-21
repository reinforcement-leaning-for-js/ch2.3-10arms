# Reinforcement Learning: An Introduction, 2nd edition
## Chapter 2 - Multi Armed Bandit / Epsilon-greedy
### Environment
Developed and tested on Ubuntu/x86-64 machine.  
Python 3.8.  
If you use Intel x86 processor, we recommend activating SVML by installing icc_rt for better performance.  
To intall, type the following command.
```conda install -c numba icc_rt```

### Stationary Case
- Each arm *i* return followes a distribution *norm(q(i), 1)* where *q(i)* is sampled from *norm(1, 0)*. 
- ```max_ensemble``` times of independent simulations are conducted and averaged.
- Each simulation goes until reach ```max_iter``` timestep.
- The mean return value of each arm does not change over time.
- Epsilon-greedy policy is implemented to find best epsilon.

### Non-stationary Case
- Initially, each arm *i* return followes a distribution *norm(q(i), 1)* where *q(i)* is sampled from *norm(1, 0)*. 
- Each *q(i)* slightly changes over time by adding *x(i)_t\~norm(0, 0.01)* on each timestep.
- Fixed-weight (exponential recency) average algorithm is also implemented to compare with arithmatic average algorithm. The weighting factor is ```alpha```.


