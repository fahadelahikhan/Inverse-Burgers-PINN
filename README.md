# Inverse-Burgers-PINN
A pure-PyTorch PINN that recovers the unknown coefficients of Burgers' equation u_t + λ1·u·u_x − λ2·u_xx = 0 (true λ1 = 1, λ2 = 0.01/π ≈ 0.003183) from 2000 sparse, noisy measurements.
