import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Use a nice style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'figure.figsize': (12, 8)
})

def theoretical_waiting_time_survival(t, lambda_rate, r, mu):
    """
    Calculate the theoretical P(W > t) for M/Er/1 queue using a more accurate approach.
    """
    rho = lambda_rate * r / mu
    if rho >= 1:
        raise ValueError("Utilization factor must be less than 1 for stability")
    
    def polynomial(z):
        return z**r - (lambda_rate/mu)*(1-z)
    
    roots = []
    initial_guesses = np.linspace(0.01, 0.99, 2*r)
    for guess in initial_guesses:
        root = fsolve(polynomial, guess)[0]
        if 0 < root < 1 and not any(abs(root-existing) < 1e-8 for existing in roots):
            roots.append(root)
    roots.sort()
    
    # Use the dominant root (closest to 1) for the approximation
    if len(roots) > 0:
        dominant_root = roots[-1]
        coefficient = rho * (1 - dominant_root) / (r - (r-1)*dominant_root)
        return coefficient * np.exp(-mu*(1-dominant_root)*t)
    else:
        return rho * np.exp(-mu*(1-np.sqrt(rho))*t)

def run_mer1_simulation(lambda_rate=0.3, r=3, mu=1.0, 
                         num_customers=100000, warm_up_customers=10000, 
                         random_seed=42):
    """
    Run the M/Er/1 simulation and collect waiting times.
    """
    np.random.seed(random_seed)
    rho = lambda_rate * r / mu
    if rho >= 1:
        raise ValueError(f"Utilization ρ = {rho} ≥ 1. The queue will be unstable.")
    
    interarrivals = np.random.exponential(1/lambda_rate, num_customers)
    service_times = np.sum(np.random.exponential(1/mu, size=(num_customers, r)), axis=1)
    
    arrivals = np.zeros(num_customers)
    departures = np.zeros(num_customers)
    service_starts = np.zeros(num_customers)
    
    arrivals[0] = interarrivals[0]
    for i in range(1, num_customers):
        arrivals[i] = arrivals[i-1] + interarrivals[i]
    
    service_starts[0] = arrivals[0]
    departures[0] = service_starts[0] + service_times[0]
    for i in range(1, num_customers):
        service_starts[i] = max(arrivals[i], departures[i-1])
        departures[i] = service_starts[i] + service_times[i]
    
    waiting_times = service_starts - arrivals
    waiting_times = waiting_times[warm_up_customers:]
    
    theo_service_time = r / mu
    theo_waiting_time = (rho * (r + 1)) / (2 * mu * (1 - rho))
    theo_sojourn_time = theo_waiting_time + theo_service_time
    
    print("\n=== Queue Statistics ===")
    print(f"Parameters: λ = {lambda_rate}, r = {r}, μ = {mu}, ρ = {rho:.4f}")
    print(f"Theoretical mean waiting time: {theo_waiting_time:.4f}")
    print(f"Simulated mean waiting time: {np.mean(waiting_times):.4f}")
    print(f"Variance of waiting time: {np.var(waiting_times):.4f}")
    
    return waiting_times, rho, theo_waiting_time

def fit_distributions_to_waiting_times(waiting_times, lambda_rate, r, mu):
    """
    Fit multiple distributions to waiting time data and compare them.
    """
    rho = lambda_rate * r / mu
    t_values = np.linspace(0, max(waiting_times), 1000)
    p_empirical = np.array([np.mean(waiting_times > t) for t in t_values])
    p_theoretical = theoretical_waiting_time_survival(t_values, lambda_rate, r, mu)
    
    distributions = [
        ('exponential', stats.expon),
        ('gamma', stats.gamma),
        ('weibull', stats.weibull_min),
        ('lognormal', stats.lognorm)
    ]
    
    results = []
    
    # Plot survival function comparison with improved styling
    plt.figure()
    plt.semilogy(t_values, p_empirical, 'b-', label='Simulation', marker='o', markevery=100)
    plt.semilogy(t_values, p_theoretical, 'r--', label='Theoretical Approximation')
    
    for dist_name, distribution in distributions:
        params = distribution.fit(waiting_times)
        log_lik = np.sum(distribution.logpdf(waiting_times, *params))
        k = len(params)
        aic = 2 * k - 2 * log_lik
        n = len(waiting_times)
        bic = k * np.log(n) - 2 * log_lik
        ks_stat, ks_p = stats.kstest(waiting_times, lambda x: distribution.cdf(x, *params))
        
        results.append({
            'name': dist_name,
            'distribution': distribution,
            'params': params,
            'aic': aic,
            'bic': bic,
            'ks_stat': ks_stat,
            'ks_p': ks_p
        })
        
        p_fitted = 1 - distribution.cdf(t_values, *params)
        plt.semilogy(t_values, p_fitted, '--', label=f'{dist_name} (AIC={aic:.2f})')
    
    plt.xlabel('Waiting Time (t)')
    plt.ylabel('P(W > t)')
    plt.title(f'Waiting Time Survival Function for M/E{r}/1 Queue (ρ = {rho:.2f})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    results.sort(key=lambda x: x['aic'])
    
    print("\n=== Distribution Fitting Results ===")
    print("Ranked by AIC (lower is better):")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['name']}: AIC: {result['aic']:.2f}, BIC: {result['bic']:.2f}, " 
              f"KS statistic: {result['ks_stat']:.4f}, p-value: {result['ks_p']:.4f}")
        print(f"   Parameters: {result['params']}\n")
    
    best_dist = results[0]
    
    plt.figure()
    hist, bins, _ = plt.hist(waiting_times, bins=30, density=True, alpha=0.5, 
                             color='gray', label='Simulation Data')
    x = np.linspace(0, max(waiting_times), 1000)
    best_pdf = best_dist['distribution'].pdf(x, *best_dist['params'])
    plt.plot(x, best_pdf, 'r-', lw=2, label=f'Best Fit: {best_dist["name"]} (AIC={best_dist["aic"]:.2f})')
    
    # Method-of-moments gamma fit
    mom_shape = np.mean(waiting_times)**2 / np.var(waiting_times)
    mom_scale = np.var(waiting_times) / np.mean(waiting_times)
    mom_pdf = stats.gamma.pdf(x, a=mom_shape, scale=mom_scale)
    plt.plot(x, mom_pdf, 'g--', lw=2, label=f'Gamma MoM (shape={mom_shape:.4f}, scale={mom_scale:.4f})')
    
    plt.xlabel('Waiting Time')
    plt.ylabel('Probability Density')
    plt.title(f'Waiting Time PDF Fitting for M/E{r}/1 Queue')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    return results, mom_shape, mom_scale

def analyze_mer1_waiting_times(lambda_rate=0.3, r=3, mu=1.0, 
                               num_customers=100000, warm_up=10000,
                               random_seed=42):
    """
    Complete analysis of M/Er/1 queue waiting times.
    """
    print(f"Analyzing M/E{r}/1 queue with λ={lambda_rate}, μ={mu}")
    
    waiting_times, rho, theo_mean = run_mer1_simulation(
        lambda_rate=lambda_rate,
        r=r,
        mu=mu,
        num_customers=num_customers,
        warm_up_customers=warm_up,
        random_seed=random_seed
    )
    
    results, mom_shape, mom_scale = fit_distributions_to_waiting_times(
        waiting_times, lambda_rate, r, mu
    )
    
    t_values = np.linspace(0, np.percentile(waiting_times, 99), 100)
    p_empirical = np.array([np.mean(waiting_times > t) for t in t_values])
    p_theoretical = theoretical_waiting_time_survival(t_values, lambda_rate, r, mu)
    best_dist = results[0]
    p_best_fit = 1 - best_dist['distribution'].cdf(t_values, *best_dist['params'])
    
    mse_theo = np.mean((p_empirical - p_theoretical)**2)
    mse_fit = np.mean((p_empirical - p_best_fit)**2)
    
    print("\n=== Approximation Accuracy ===")
    print(f"Mean squared error of theoretical approximation: {mse_theo:.8f}")
    print(f"Mean squared error of best fit distribution: {mse_fit:.8f}")
    
    return {
        'waiting_times': waiting_times,
        'rho': rho,
        'theoretical_mean': theo_mean,
        'empirical_mean': np.mean(waiting_times),
        'best_distribution': best_dist['name'],
        'best_dist_params': best_dist['params'],
        'mom_gamma_shape': mom_shape,
        'mom_gamma_scale': mom_scale,
        'mse_theoretical': mse_theo,
        'mse_best_fit': mse_fit
    }

# Run the complete analysis with default parameters
results = analyze_mer1_waiting_times(
    lambda_rate=0.3,  # Arrival rate
    r=3,              # Number of Erlang phases
    mu=1.0,           # Service rate per phase
    num_customers=100000,
    warm_up=10000,
    random_seed=42
)
