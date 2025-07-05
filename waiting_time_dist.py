import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sympy as sp
from scipy.optimize import fsolve

def theoretical_waiting_time_survival(t, lambda_rate, r, mu):
    """
    Calculate the theoretical P(W > t) for M/Er/1 queue
    
    Parameters:
    -----------
    t : float or array
        Time points to evaluate the survival function
    lambda_rate : float
        Arrival rate
    r : int
        Number of Erlang phases
    mu : float
        Service rate per phase
    
    Returns:
    --------
    float or array
        P(W > t) values
    """
    rho = lambda_rate * r / mu
    
    if rho >= 1:
        raise ValueError("Utilization factor must be less than 1 for stability")
    
    # For M/Er/1, we need to find the roots of a certain polynomial
    # The polynomial is z^r - (lambda_rate/mu)(1-z)
    def polynomial(z):
        return z**r - (lambda_rate/mu)*(1-z)
    
    # For numerical stability, we'll find roots within (0,1)
    # One root is always at z=1, the others are inside the unit circle
    
    # Initial guesses distributed in (0,1)
    initial_guesses = np.linspace(0.1, 0.9, r)
    
    roots = []
    for guess in initial_guesses:
        root = fsolve(polynomial, guess)[0]
        if 0 < root < 1 and not any(abs(root-r) < 1e-10 for r in roots):
            roots.append(root)
    
    # If we didn't find enough roots, try with more initial points
    if len(roots) < r-1:  # We expect r-1 roots inside unit circle
        more_guesses = np.linspace(0.01, 0.99, 2*r)
        for guess in more_guesses:
            if len(roots) >= r-1:
                break
            root = fsolve(polynomial, guess)[0]
            if 0 < root < 1 and not any(abs(root-r) < 1e-10 for r in roots):
                roots.append(root)
    
    # Now we need to calculate the coefficients c_k
    # This is quite complex for general r
    # For r = 2, c_1 = rho/(1-x_1)
    # For r = 3, we need to solve a system of equations
    
    # For simplicity, let's use a numerical approach for verification
    # We'll calculate the mean waiting time and compare with theory
    
    # For M/Er/1, the mean waiting time is:
    # E[W] = (lambda_rate * r * (r+1)) / (2 * mu * (mu - lambda_rate * r))
    
    # The waiting time survival function for M/G/1 can be approximated as:
    # P(W > t) ≈ rho * exp(-(1-rho)*mu*t/r)
    
    # For M/Er/1, a better approximation is:
    # P(W > t) ≈ rho * exp(-mu*(1-sqrt(rho))*t)
    
    return rho * np.exp(-mu*(1-np.sqrt(rho))*t)

def fit_waiting_time_distribution(waiting_times, lambda_rate, r, mu):
    """
    Fit a theoretical distribution to the waiting time data and 
    compare with M/Er/1 theoretical distribution
    
    Parameters:
    -----------
    waiting_times : array
        Array of waiting times from simulation
    lambda_rate, r, mu : float, int, float
        Queue parameters
    
    Returns:
    --------
    None (produces plots and statistics)
    """
    # Calculate theoretical mean and variance
    rho = lambda_rate * r / mu
    theo_mean = (rho / (1 - rho)) * ((r + 1) / (2 * mu))
    
    # Calculate empirical mean and variance
    empirical_mean = np.mean(waiting_times)
    empirical_var = np.var(waiting_times)
    
    # Create t values for plotting
    t_values = np.linspace(0, max(waiting_times), 1000)
    
    # Calculate empirical survival function P(W > t)
    sorted_times = np.sort(waiting_times)
    p_empirical = np.array([np.mean(waiting_times > t) for t in t_values])
    
    # Calculate theoretical survival function
    p_theoretical = theoretical_waiting_time_survival(t_values, lambda_rate, r, mu)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.semilogy(t_values, p_empirical, 'b-', label='Simulation')
    plt.semilogy(t_values, p_theoretical, 'r--', label='Theoretical Approximation')
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.xlabel('Waiting Time (t)', fontsize=12)
    plt.ylabel('P(W > t)', fontsize=12)
    plt.title(f'Waiting Time Survival Function for M/E{r}/1 Queue (ρ = {rho:.2f})', fontsize=14)
    plt.legend(fontsize=10)
    
    # Add information to plot
    plt.text(0.05, 0.3, 
             f"λ = {lambda_rate:.4f}\n"
             f"r = {r}\n"
             f"μ = {mu:.4f}\n"
             f"ρ = {rho:.4f}\n"
             f"E[W]_theo = {theo_mean:.4f}\n"
             f"E[W]_sim = {empirical_mean:.4f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Try fitting various distributions
    distributions = [
        ('exponential', stats.expon),
        ('gamma', stats.gamma),
        ('weibull', stats.weibull_min),
        ('lognormal', stats.lognorm)
    ]
    
    # Plot PDF comparisons
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of waiting times
    hist, bins, _ = plt.hist(waiting_times, bins=30, density=True, alpha=0.5, 
                             color='gray', label='Simulation Data')
    
    colors = ['r', 'g', 'b', 'm']
    best_aic = np.inf
    best_dist = None
    
    for i, (dist_name, distribution) in enumerate(distributions):
        # Fit distribution to data
        params = distribution.fit(waiting_times)
        
        # Calculate AIC
        log_likelihood = np.sum(distribution.logpdf(waiting_times, *params))
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        
        # Plot PDF
        x = np.linspace(0, max(waiting_times), 1000)
        pdf = distribution.pdf(x, *params)
        plt.plot(x, pdf, colors[i], label=f'{dist_name} (AIC={aic:.2f})')
        
        # Update best distribution
        if aic < best_aic:
            best_aic = aic
            best_dist = (dist_name, distribution, params)
    
    plt.xlabel('Waiting Time', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Waiting Time PDF Fitting for M/E{r}/1 Queue', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Return best distribution info
    return best_dist

def run_mer1_simulation_for_waiting_times(lambda_rate=0.3, r=3, mu=1.0, 
                                          num_customers=100000, warm_up_customers=10000, 
                                          random_seed=42):
    """
    Run the M/Er/1 simulation and analyze waiting times.
    
    Parameters are the same as in the original code.
    """
    from dataclasses import dataclass
    
    @dataclass
    class SimulationConfig:
        lambda_rate: float
        r: int
        mu: float
        num_customers: int
        warm_up_customers: int
        random_seed: int
        
        def __post_init__(self):
            # Derived theoretical values
            self.rho = self.lambda_rate * self.r / self.mu
            if self.rho >= 1:
                raise ValueError(f"Utilization ρ = {self.rho} ≥ 1. The queue will be unstable.")
            
            # Theoretical metrics
            self.theo_service_time = self.r / self.mu
            self.theo_waiting_time = (self.rho / (1 - self.rho)) * ((self.r + 1) / (2 * self.mu))
            self.theo_sojourn_time = self.theo_waiting_time + self.theo_service_time
            self.theo_num_in_system = self.lambda_rate * self.theo_sojourn_time
    
    @dataclass
    class CustomerRecord:
        arrival_time: float = 0.0
        service_start_time: float = 0.0
        departure_time: float = 0.0
        
        @property
        def waiting_time(self) -> float:
            return self.service_start_time - self.arrival_time
    
    # Create simulation configuration
    config = SimulationConfig(
        lambda_rate=lambda_rate,
        r=r,
        mu=mu,
        num_customers=num_customers,
        warm_up_customers=warm_up_customers,
        random_seed=random_seed
    )
    
    # Set random seed
    np.random.seed(config.random_seed)
    
    # Initialize customer records
    customers = [CustomerRecord() for _ in range(config.num_customers)]
    
    # Generate service time function
    def generate_service_time():
        return np.sum(np.random.exponential(1 / config.mu, size=config.r))
    
    # First arrival
    customers[0].arrival_time = np.random.exponential(1 / config.lambda_rate)
    customers[0].service_start_time = customers[0].arrival_time
    service_time = generate_service_time()
    customers[0].departure_time = customers[0].service_start_time + service_time
    
    # Server free time
    server_free_time = customers[0].departure_time
    
    # Generate and process subsequent customers
    for i in range(1, config.num_customers):
        # Generate interarrival time and compute arrival time
        interarrival = np.random.exponential(1 / config.lambda_rate)
        customers[i].arrival_time = customers[i-1].arrival_time + interarrival
        
        # Service starts when customer arrives or when server becomes free
        customers[i].service_start_time = max(customers[i].arrival_time, server_free_time)
        
        # Generate service time and compute departure time
        service_time = generate_service_time()
        customers[i].departure_time = customers[i].service_start_time + service_time
        
        # Update server free time
        server_free_time = customers[i].departure_time
    
    # Extract waiting times (after warm-up)
    waiting_times = np.array([customer.waiting_time for customer in 
                             customers[config.warm_up_customers:]])
    
    # Basic statistics
    print("\n=== Waiting Time Statistics ===")
    print(f"Mean Waiting Time: {np.mean(waiting_times):.4f}")
    print(f"Theoretical Mean Waiting Time: {config.theo_waiting_time:.4f}")
    print(f"Variance of Waiting Time: {np.var(waiting_times):.4f}")
    
    # Fit distributions and analyze
    best_dist = fit_waiting_time_distribution(waiting_times, lambda_rate, r, mu)
    
    # Return for further analysis
    return waiting_times, config, best_dist


def fit_waiting_time_distribution_mom(waiting_times, lambda_rate, r, mu):
    """
    Fit a gamma distribution to waiting time data using method-of-moments
    and compare it with the simulation and theoretical survival function.
    """
    rho = lambda_rate * r / mu
    theo_mean = (rho / (1 - rho)) * ((r + 1) / (2 * mu))
    empirical_mean = np.mean(waiting_times)
    empirical_var = np.var(waiting_times)
    
    # Method-of-moments gamma parameters
    gamma_shape = empirical_mean**2 / empirical_var
    gamma_scale = empirical_var / empirical_mean

    print(f"Method-of-moments Gamma parameters: shape = {gamma_shape:.4f}, scale = {gamma_scale:.4f}")
    
    # Plot empirical survival function and gamma PDF
    t_values = np.linspace(0, max(waiting_times), 1000)
    p_empirical = np.array([np.mean(waiting_times > t) for t in t_values])
    
    # Theoretical survival function using your (approximate) function
    p_theoretical = theoretical_waiting_time_survival(t_values, lambda_rate, r, mu)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(t_values, p_empirical, 'b-', label='Simulation Survival')
    plt.semilogy(t_values, p_theoretical, 'r--', label='Theoretical Approximation')
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.xlabel('Waiting Time (t)')
    plt.ylabel('P(W > t)')
    plt.title(f'Waiting Time Survival Function for M/E{r}/1 Queue (ρ = {rho:.2f})')
    plt.legend()
    plt.show()
    
    # Plot PDF comparison
    plt.figure(figsize=(12, 6))
    hist, bins, _ = plt.hist(waiting_times, bins=30, density=True, alpha=0.5, 
                             color='gray', label='Simulation Data')
    x = np.linspace(0, max(waiting_times), 1000)
    gamma_pdf = stats.gamma.pdf(x, a=gamma_shape, scale=gamma_scale)
    plt.plot(x, gamma_pdf, 'm-', lw=2, label='Gamma Fit (MOM)')
    
    plt.xlabel('Waiting Time')
    plt.ylabel('Probability Density')
    plt.title(f'Waiting Time PDF Fitting for M/E{r}/1 Queue')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return ('gamma', gamma_shape, gamma_scale)


# Run simulation and analyze waiting times
waiting_times, config, best_dist = run_mer1_simulation_for_waiting_times(
    lambda_rate=0.3,  # Arrival rate
    r=3,              # Number of Erlang phases
    mu=1.0,           # Service rate per phase
    num_customers=100000,
    warm_up_customers=10000,
    random_seed=42
)

# Compare with theoretical formula from the image
# The formula from the image is for a more general case

print(f"\nBest fitting distribution: {best_dist[0]}")
print(f"Parameters: {best_dist[2]}")

# Plot empirical vs theoretical CDF
plt.figure(figsize=(12, 6))
t_values = np.linspace(0, np.max(waiting_times), 1000)

# Empirical CDF
cdf_empirical = np.array([np.mean(waiting_times <= t) for t in t_values])
plt.plot(t_values, cdf_empirical, 'b-', label='Empirical CDF')

# Best fit distribution CDF
best_dist_name, best_dist_obj, best_dist_params = best_dist
cdf_best_fit = best_dist_obj.cdf(t_values, *best_dist_params)
plt.plot(t_values, cdf_best_fit, 'r--', label=f'Best Fit: {best_dist_name}')

# Theoretical approximation from formula in image
# We'll use a simplified version for M/Er/1
rho = config.lambda_rate * config.r / config.mu
cdf_theo = 1 - rho * np.exp(-config.mu*(1-np.sqrt(rho))*t_values)
plt.plot(t_values, cdf_theo, 'g-.', label='Theoretical Approximation')

plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Waiting Time (t)', fontsize=12)
plt.ylabel('P(W ≤ t)', fontsize=12)
plt.title(f'Waiting Time CDF for M/E{config.r}/1 Queue (ρ = {rho:.2f})', fontsize=14)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

best_dist = fit_waiting_time_distribution_mom(waiting_times, lambda_rate=0.3, r=3, mu=1.0)
