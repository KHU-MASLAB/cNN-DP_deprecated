# cNN-DP: Composite neural network with differential propagation for impulsive nonlinear dynamics
In mechanical engineering, abundant high-quality data from simulations and experimental observations can help develop practical and accurate data-driven models. However, when dynamics are complex and highly nonlinear, designing a suitable model and optimizing it accurately is challenging. In particular, when data comprise impulsive signals or high-frequency components, training a data-driven model becomes increasingly challenging. This study proposes a novel and robust composite neural network for impulsive time-transient dynamics by dividing the prediction of the dynamics into tasks for three sub-networks, one for approximating simplified dynamics and the other two for mapping lower-order derivatives to higher-order derivatives. The mapping serves as the temporal differential operator, hence, the name “composite neural network with differential propagation (cNN-DP)” for the suggested model. Furthermore, numerical investigations were conducted to compare cNN-DP with two baseline models, a conventional network and another employing the autogradient approach. Regarding the convergence rate of model optimizations and the generalization accuracy, the proposed network outperformed the baseline models by many orders of magnitude. In terms of computational efficiency, numerical tests showed that cNN-DP requires an acceptable and comparable computational load. Although the numerical studies and descriptions focus on accelerations, the proposed network can be easily extended to any other application involving impulsive data.
