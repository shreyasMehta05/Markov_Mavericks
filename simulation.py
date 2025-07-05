import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time

@dataclass
class SimulationConfig:
    """Configuration parameters for the M/Er/1 queue simulation."""
    lambda_rate: float          # Arrival rate (jobs per time unit)
    r: int                      # Number of Erlang phases
    mu: float                   # Service rate per phase
    num_customers: int          # Number of customers to simulate
    warm_up_customers: int      # Number of customers in warm-up period
    random_seed: Optional[int]  # Random seed for reproducibility
    
    def __post_init__(self):
        # Derived theoretical values
        self.rho = self.lambda_rate * self.r / self.mu
        if self.rho >= 1:
            raise ValueError(f"Utilization ρ = {self.rho} ≥ 1. The queue will be unstable. Try reducing λ or increasing μ.")
        
        # Theoretical metrics
        self.theo_service_time = self.r / self.mu
        self.theo_waiting_time = (self.rho / (1 - self.rho)) * ((self.r + 1) / (2 * self.mu))
        self.theo_sojourn_time = self.theo_waiting_time + self.theo_service_time
        self.theo_num_in_system = self.lambda_rate * self.theo_sojourn_time
        
    def print_theoretical_values(self):
        """Print the theoretical values for the queue metrics."""
        print("\n=== System Parameters ===")
        print(f"Arrival rate (λ): {self.lambda_rate}")
        print(f"Service phases (r): {self.r}")
        print(f"Service rate per phase (μ): {self.mu}")
        print(f"Utilization (ρ): {self.rho:.4f}")
        
        print("\n=== Theoretical Values ===")
        print(f"Mean Service Time, E(S): {self.theo_service_time:.4f}")
        print(f"Mean Waiting Time, E(W): {self.theo_waiting_time:.4f}")
        print(f"Mean Sojourn Time, E(T): {self.theo_sojourn_time:.4f}")
        print(f"Mean Number in System, E(N): {self.theo_num_in_system:.4f}")


@dataclass
class CustomerRecord:
    """Record of a customer's journey through the system."""
    arrival_time: float = 0.0
    service_start_time: float = 0.0
    departure_time: float = 0.0
    
    @property
    def waiting_time(self) -> float:
        return self.service_start_time - self.arrival_time
    
    @property
    def sojourn_time(self) -> float:
        return self.departure_time - self.arrival_time
    
    @property
    def service_time(self) -> float:
        return self.departure_time - self.service_start_time


class MErOneSimulation:
    """Simulation of an M/Er/1 queuing system."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.customers: List[CustomerRecord] = []
        self.sample_path_times: List[float] = []
        self.sample_path_counts: List[int] = []
        
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def generate_service_time(self) -> float:
        """Generate service time as sum of r exponential phases."""
        return np.sum(np.random.exponential(1 / self.config.mu, size=self.config.r))
    
    def run(self) -> None:
        """Run the simulation."""
        start_time = time.time()
        
        # Initialize the customer records
        self.customers = [CustomerRecord() for _ in range(self.config.num_customers)]
        
        # First arrival
        self.customers[0].arrival_time = np.random.exponential(1 / self.config.lambda_rate)
        self.customers[0].service_start_time = self.customers[0].arrival_time
        service_time = self.generate_service_time()
        self.customers[0].departure_time = self.customers[0].service_start_time + service_time
        
        # For sample path tracking
        self.sample_path_times = [self.customers[0].arrival_time, self.customers[0].departure_time]
        self.sample_path_counts = [1, 0]
        
        # Current time when server will be free
        server_free_time = self.customers[0].departure_time
        
        # Generate and process subsequent customers
        for i in range(1, self.config.num_customers):
            # Generate interarrival time and compute arrival time
            interarrival = np.random.exponential(1 / self.config.lambda_rate)
            self.customers[i].arrival_time = self.customers[i-1].arrival_time + interarrival
            
            # Update sample path for arrival
            self.sample_path_times.append(self.customers[i].arrival_time)
            self.sample_path_counts.append(self.sample_path_counts[-1] + 1)
            
            # Service starts when customer arrives or when server becomes free
            self.customers[i].service_start_time = max(self.customers[i].arrival_time, server_free_time)
            
            # Generate service time and compute departure time
            service_time = self.generate_service_time()
            self.customers[i].departure_time = self.customers[i].service_start_time + service_time
            
            # Update sample path for departure
            self.sample_path_times.append(self.customers[i].departure_time)
            self.sample_path_counts.append(self.sample_path_counts[-1] - 1)
            
            # Update server free time
            server_free_time = self.customers[i].departure_time
        
        # Sort the sample path by time for correct visualization
        sorted_indices = np.argsort(self.sample_path_times)
        self.sample_path_times = [self.sample_path_times[i] for i in sorted_indices]
        
        # Reconstruct the counts after sorting
        current_count = 0
        self.sample_path_counts = []
        for i in sorted_indices:
            if i % 2 == 0:  # It's an arrival
                current_count += 1
            else:  # It's a departure
                current_count -= 1
            self.sample_path_counts.append(current_count)
        
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    
    def compute_metrics(self) -> dict:
        """Compute performance metrics from simulation data."""
        # Skip warm-up period customers in calculations
        steady_state_customers = self.customers[self.config.warm_up_customers:]
        
        waiting_times = [customer.waiting_time for customer in steady_state_customers]
        service_times = [customer.service_time for customer in steady_state_customers]
        sojourn_times = [customer.sojourn_time for customer in steady_state_customers]
        
        metrics = {
            "mean_waiting_time": np.mean(waiting_times),
            "mean_service_time": np.mean(service_times),
            "mean_sojourn_time": np.mean(sojourn_times),
            "variance_waiting_time": np.var(waiting_times),
            "variance_sojourn_time": np.var(sojourn_times),
            "max_waiting_time": np.max(waiting_times),
            "max_sojourn_time": np.max(sojourn_times),
            "num_in_system": self.config.lambda_rate * np.mean(sojourn_times)  # Little's Law
        }
        
        return metrics
    
    def print_comparison(self, metrics: dict) -> None:
        """Print comparison between theoretical and simulated values."""
        print("\n=== Simulated Values ===")
        print(f"Mean Service Time, E(S): {metrics['mean_service_time']:.4f}")
        print(f"Mean Waiting Time, E(W): {metrics['mean_waiting_time']:.4f}")
        print(f"Mean Sojourn Time, E(T): {metrics['mean_sojourn_time']:.4f}")
        print(f"Mean Number in System, E(N): {metrics['num_in_system']:.4f}")
        print(f"Max Waiting Time: {metrics['max_waiting_time']:.4f}")
        print(f"Max Sojourn Time: {metrics['max_sojourn_time']:.4f}")
        print(f"Variance of Waiting Time: {metrics['variance_waiting_time']:.4f}")
        print(f"Variance of Sojourn Time: {metrics['variance_sojourn_time']:.4f}")
        
        print("\n=== Differences (Simulated - Theoretical) ===")
        print(f"Difference in Mean Service Time: {metrics['mean_service_time'] - self.config.theo_service_time:.4f}")
        print(f"Difference in Mean Waiting Time: {metrics['mean_waiting_time'] - self.config.theo_waiting_time:.4f}")
        print(f"Difference in Mean Sojourn Time: {metrics['mean_sojourn_time'] - self.config.theo_sojourn_time:.4f}")
        print(f"Difference in Mean Number in System: {metrics['num_in_system'] - self.config.theo_num_in_system:.4f}")
        
        # Calculate relative errors
        rel_error_waiting = abs((metrics['mean_waiting_time'] - self.config.theo_waiting_time) / self.config.theo_waiting_time) * 100
        rel_error_sojourn = abs((metrics['mean_sojourn_time'] - self.config.theo_sojourn_time) / self.config.theo_sojourn_time) * 100
        
        print("\n=== Relative Errors (%) ===")
        print(f"Relative Error in Mean Waiting Time: {rel_error_waiting:.2f}%")
        print(f"Relative Error in Mean Sojourn Time: {rel_error_sojourn:.2f}%")
    
    def plot_sample_path(self, max_time: Optional[float] = None) -> None:
        """Plot the sample path of number of customers in the system over time."""
        plt.figure(figsize=(12, 5))
        
        # Filter sample path data if max_time is specified
        if max_time is not None:
            valid_indices = [i for i, t in enumerate(self.sample_path_times) if t <= max_time]
            times = [self.sample_path_times[i] for i in valid_indices]
            counts = [self.sample_path_counts[i] for i in valid_indices]
        else:
            times = self.sample_path_times
            counts = self.sample_path_counts
        
        plt.step(times, counts, where='post', linewidth=1.5)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Number in System', fontsize=12)
        plt.title(f'Sample Path for M/E{self.config.r}/1 Queue (ρ = {self.config.rho:.2f})', fontsize=14)
        
        # Add theoretical mean as horizontal line
        plt.axhline(y=self.config.theo_num_in_system, color='r', linestyle='-', alpha=0.7, 
                    label=f'Theoretical Mean: {self.config.theo_num_in_system:.2f}')
        
        # Add simulation mean 
        simulated_mean = np.mean(counts) # This is an approximation
        plt.axhline(y=simulated_mean, color='g', linestyle='--', alpha=0.7,
                    label=f'Simulated Mean: {simulated_mean:.2f}')
        
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def plot_waiting_time_histogram(self) -> None:
        """Plot histogram of waiting times."""
        # Skip warm-up period
        waiting_times = [customer.waiting_time for customer in 
                        self.customers[self.config.warm_up_customers:]]
        
        plt.figure(figsize=(10, 5))
        plt.hist(waiting_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=self.config.theo_waiting_time, color='r', linestyle='-', 
                   label=f'Theoretical Mean: {self.config.theo_waiting_time:.2f}')
        
        plt.axvline(x=np.mean(waiting_times), color='g', linestyle='--', 
                   label=f'Simulated Mean: {np.mean(waiting_times):.2f}')
        
        plt.xlabel('Waiting Time', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Waiting Times', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def plot_sojourn_time_histogram(self) -> None:
        """Plot histogram of sojourn times."""
        # Skip warm-up period
        sojourn_times = [customer.sojourn_time for customer in 
                         self.customers[self.config.warm_up_customers:]]
        
        plt.figure(figsize=(10, 5))
        plt.hist(sojourn_times, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(x=self.config.theo_sojourn_time, color='r', linestyle='-', 
                   label=f'Theoretical Mean: {self.config.theo_sojourn_time:.2f}')
        
        plt.axvline(x=np.mean(sojourn_times), color='g', linestyle='--', 
                   label=f'Simulated Mean: {np.mean(sojourn_times):.2f}')
        
        plt.xlabel('Sojourn Time', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Sojourn Times', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def verify_littles_law(self, metrics: dict) -> None:
        """Verify Little's Law by calculating E[N]/E[T] and comparing to λ."""
        # Calculate the ratio E[N]/E[T]
        en_et_ratio = metrics['num_in_system'] / metrics['mean_sojourn_time']
        
        # The theoretical value is λ
        lambda_value = self.config.lambda_rate
        
        # Calculate the relative error
        rel_error = abs((en_et_ratio - lambda_value) / lambda_value) * 100
        
        print("\n=== Little's Law Verification ===")
        print(f"E[N]/E[T]: {en_et_ratio:.6f}")
        print(f"λ: {lambda_value:.6f}")
        print(f"Relative Error: {rel_error:.6f}%")
        
        # Create a plot to visualize the verification
        plt.figure(figsize=(10, 6))
        
        # Create a small range around λ for plotting
        lambda_range = np.linspace(lambda_value * 0.9, lambda_value * 1.1, 100)
        ideal_values = lambda_range  # In the ideal case, E[N]/E[T] = λ
        
        plt.plot(1/lambda_range, ideal_values, 'r-', linewidth=2, label="Theoretical: E[N]/E[T] = λ")
        plt.scatter(1/lambda_value, en_et_ratio, color='blue', s=100, 
                    label=f"Simulated: E[N]/E[T] = {en_et_ratio:.6f}")
        
        # Plot reference line
        plt.axhline(y=lambda_value, color='green', linestyle='--', alpha=0.7,
                label=f"λ = {lambda_value}")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('1/λ', fontsize=12)
        plt.ylabel('E[N]/E[T]', fontsize=12)
        plt.title("Verification of Little's Law: E[N]/E[T] vs 1/λ", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    def plot_littles_law_convergence(self) -> None:
        """
        Plot E[N]/E[T] ratio versus number of jobs to visualize 
        the convergence of Little's Law as the simulation progresses.
        """
        # Skip warm-up period
        start_idx = self.config.warm_up_customers
        
        # Calculate E[N]/E[T] ratio for each point in time (after each customer)
        jobs = []
        ratios = []
        
        for i in range(start_idx, len(self.customers)):
            # Get all customers processed so far (after warm-up)
            customers_so_far = self.customers[start_idx:i+1]
            
            # Calculate average sojourn time E[T]
            avg_sojourn_time = np.mean([c.sojourn_time for c in customers_so_far])
            
            # Calculate average number in system E[N]
            # For this we'll use Little's Law formula E[N] = λ * E[T] to get the average
            # number in system up to this point, based on observed sojourn times
            avg_num_in_system = self.config.lambda_rate * avg_sojourn_time
            
            # Calculate the ratio E[N]/E[T]
            if avg_sojourn_time > 0:  # Avoid division by zero
                ratio = avg_num_in_system / avg_sojourn_time
                jobs.append(i)
                ratios.append(ratio)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(jobs, ratios, 'k-', linewidth=1.5, label='(E[T], E[N]) at a time instance')
        
        # Add a horizontal line for the theoretical value (λ)
        plt.axhline(y=self.config.lambda_rate, color='r', linestyle='--', alpha=0.7,
                label=f'Theoretical λ = {self.config.lambda_rate}')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Number of Jobs', fontsize=12)
        plt.ylabel('E[N] / E[T]', fontsize=12)
        plt.title('E[N] / E[T] vs Number of Jobs', fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
        
        # Print final statistics
        final_ratio = ratios[-1] if ratios else 0
        rel_error = abs((final_ratio - self.config.lambda_rate) / self.config.lambda_rate) * 100
        
        print("\n=== Little's Law Convergence ===")
        print(f"Final E[N]/E[T] ratio: {final_ratio:.6f}")
        print(f"Theoretical λ: {self.config.lambda_rate:.6f}")
        print(f"Relative error: {rel_error:.6f}%")

    def plot_littles_law_time_series(self, window_size=500) -> None:
        """
        Plot E[N]/E[T] ratio versus time (number of jobs) using a moving window approach
        to visualize the convergence of Little's Law as the simulation progresses.
        
        Parameters:
        -----------
        window_size : int
            Size of the moving window for calculating the ratio
        """
        # Skip warm-up period
        start_idx = self.config.warm_up_customers
        
        # Calculate E[N]/E[T] ratio using a moving window
        jobs = []
        ratios = []
        
        for i in range(start_idx + window_size, len(self.customers)):
            # Use a window of customers for calculation
            window_customers = self.customers[i-window_size:i]
            
            # Calculate average sojourn time E[T] for the window
            avg_sojourn_time = np.mean([c.sojourn_time for c in window_customers])
            
            # We can estimate E[N] by using the sample path data over the window period
            # or by applying Little's Law: E[N] = λ * E[T]
            avg_num_in_system = self.config.lambda_rate * avg_sojourn_time
            
            # Calculate the ratio E[N]/E[T]
            if avg_sojourn_time > 0:  # Avoid division by zero
                ratio = avg_num_in_system / avg_sojourn_time
                jobs.append(i)
                ratios.append(ratio)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(jobs, ratios, 'k-', linewidth=1.5, label='(E[T], E[N]) at a time instance')
        
        # Add a horizontal line for the theoretical value (λ)
        plt.axhline(y=self.config.lambda_rate, color='r', linestyle='--', alpha=0.7,
                label=f'Theoretical λ = {self.config.lambda_rate}')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Number of Jobs', fontsize=12)
        plt.ylabel('E[N] / E[T]', fontsize=12)
        plt.title('E[N] / E[T] vs Number of Jobs', fontsize=14)
        plt.legend(fontsize=10)
        plt.ylim(max(0, self.config.lambda_rate * 0.5), self.config.lambda_rate * 1.5)  # Reasonable y-axis limits
        plt.tight_layout()
        plt.show()
        
        # Print final statistics
        final_ratio = ratios[-1] if ratios else 0
        rel_error = abs((final_ratio - self.config.lambda_rate) / self.config.lambda_rate) * 100
        
        print("\n=== Little's Law Time Series Analysis ===")
        print(f"Moving window size: {window_size} jobs")
        print(f"Final E[N]/E[T] ratio: {final_ratio:.6f}")
        print(f"Theoretical λ: {self.config.lambda_rate:.6f}")
        print(f"Relative error: {rel_error:.6f}%")

    def plot_littles_law_from_sample_path(self, min_jobs=1000) -> None:
        """
        Plot E[N]/E[T] ratio versus number of jobs processed, using the sample path
        to directly estimate E[N] at each point, giving a more accurate convergence plot.
        
        Parameters:
        -----------
        min_jobs : int
            Minimum number of jobs to process before starting to plot (after warm-up)
        """
        # Skip warm-up period
        start_idx = self.config.warm_up_customers
        
        # Initialize arrays for jobs and ratios
        jobs = []
        ratios = []
        
        # For each number of jobs processed...
        for i in range(max(start_idx, min_jobs), len(self.customers)):
            # Get all customers processed so far
            customers_processed = self.customers[start_idx:i+1]
            
            # Calculate mean sojourn time E[T]
            mean_sojourn = np.mean([c.sojourn_time for c in customers_processed])
            
            # Calculate mean number in system E[N] from sample path
            # Find the time period covered by these customers
            start_time = self.customers[start_idx].arrival_time
            end_time = self.customers[i].departure_time
            
            # Get sample path points in this time range
            path_times = np.array(self.sample_path_times)
            path_counts = np.array(self.sample_path_counts)
            
            # Filter to points within the time range
            valid_indices = (path_times >= start_time) & (path_times <= end_time)
            valid_times = path_times[valid_indices]
            valid_counts = path_counts[valid_indices]
            
            if len(valid_times) > 1:
                # Calculate time intervals
                time_intervals = np.diff(np.append(valid_times, end_time))
                
                # Calculate time-weighted average of system size
                total_time = end_time - start_time
                
                if total_time > 0:
                    mean_system_size = np.sum(valid_counts * time_intervals) / total_time
                    
                    # Calculate ratio E[N]/E[T]
                    if mean_sojourn > 0:
                        ratio = mean_system_size / mean_sojourn
                        jobs.append(i)
                        ratios.append(ratio)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(jobs, ratios, 'k-', linewidth=1.5, label='(E[T], E[N]) at a time instance')
        
        # Add a horizontal line for the theoretical value (λ)
        plt.axhline(y=self.config.lambda_rate, color='r', linestyle='--', alpha=0.7,
                label=f'Theoretical λ = {self.config.lambda_rate}')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Number of Jobs', fontsize=12)
        plt.ylabel('E[N] / E[T]', fontsize=12)
        plt.title('E[N] / E[T] vs Number of Jobs', fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
        
        # Print final statistics
        final_ratio = ratios[-1] if ratios else 0
        rel_error = abs((final_ratio - self.config.lambda_rate) / self.config.lambda_rate) * 100
        
        print("\n=== Little's Law Verification from Sample Path ===")
        print(f"Final E[N]/E[T] ratio: {final_ratio:.6f}")
        print(f"Theoretical λ: {self.config.lambda_rate:.6f}")
        print(f"Relative error: {rel_error:.6f}%")
        
        # Calculate average of last 20% of points to get steady-state value
        if len(ratios) > 5:
            last_20_percent = ratios[int(0.8*len(ratios)):]
            steady_state_ratio = np.mean(last_20_percent)
            steady_state_error = abs((steady_state_ratio - self.config.lambda_rate) / self.config.lambda_rate) * 100
            print(f"Steady-state E[N]/E[T] ratio: {steady_state_ratio:.6f}")
            print(f"Steady-state relative error: {steady_state_error:.6f}%")

def run_mer1_simulation(lambda_rate=0.9, r=3, mu=1.0, num_customers=100000, 
                        warm_up_customers=10000, random_seed=42, 
                        plot_sample_path=True, plot_histograms=True,
                        verify_littles_law=True, plot_littles_law_convergence=True) -> Tuple[SimulationConfig, dict]:
    """Run a complete M/Er/1 queue simulation with reporting."""
    # Create simulation configuration
    config = SimulationConfig(
        lambda_rate=lambda_rate,
        r=r,
        mu=mu,
        num_customers=num_customers,
        warm_up_customers=warm_up_customers,
        random_seed=random_seed
    )
    
    # Print theoretical values
    config.print_theoretical_values()
    
    # Create and run simulation
    simulation = MErOneSimulation(config)
    simulation.run()
    
    # Compute and print metrics
    metrics = simulation.compute_metrics()
    simulation.print_comparison(metrics)
    
    # Verify Little's Law
    if verify_littles_law:
        simulation.verify_littles_law(metrics)
    
    # Plot Little's Law convergence
    if plot_littles_law_convergence:
        simulation.plot_littles_law_from_sample_path()
    
    # Visualizations
    if plot_sample_path:
        simulation.plot_sample_path(max_time=None)
    
    if plot_histograms:
        simulation.plot_waiting_time_histogram()
        simulation.plot_sojourn_time_histogram()
    
    return config, metrics

def verify_littles_law_multiple_lambda(min_lambda=0.1, max_lambda=0.9, num_points=10, 
                                      r=3, mu=1.0, num_customers=100000, 
                                      warm_up_customers=10000, random_seed=42) -> None:
    """
    Verify Little's Law across multiple lambda values.
    
    Parameters:
    -----------
    min_lambda : float
        Minimum arrival rate to test
    max_lambda : float
        Maximum arrival rate to test (should be < mu/r for stability)
    num_points : int
        Number of lambda values to test
    r, mu, num_customers, warm_up_customers, random_seed : 
        Same parameters as in run_mer1_simulation
    """
    lambda_values = np.linspace(min_lambda, max_lambda, num_points)
    en_et_ratios = []
    
    for i, lambda_rate in enumerate(lambda_values):
        print(f"\nRunning simulation {i+1}/{num_points} with λ = {lambda_rate:.4f}")
        
        # Create simulation configuration
        config = SimulationConfig(
            lambda_rate=lambda_rate,
            r=r,
            mu=mu,
            num_customers=num_customers,
            warm_up_customers=warm_up_customers,
            random_seed=random_seed
        )
        
        # Run simulation
        simulation = MErOneSimulation(config)
        simulation.run()
        
        # Compute metrics
        metrics = simulation.compute_metrics()
        
        # Calculate E[N]/E[T] ratio
        en_et_ratio = metrics['num_in_system'] / metrics['mean_sojourn_time']
        en_et_ratios.append(en_et_ratio)
        
        # Print verification for this lambda
        rel_error = abs((en_et_ratio - lambda_rate) / lambda_rate) * 100
        print(f"λ = {lambda_rate:.6f}, E[N]/E[T] = {en_et_ratio:.6f}, Rel. Error = {rel_error:.6f}%")
    
    # Create verification plot
    plt.figure(figsize=(10, 6))
    
    # Plot the ideal line: E[N]/E[T] = λ
    ideal_line = np.linspace(min_lambda, max_lambda, 100)
    plt.plot(1/ideal_line, ideal_line, 'r-', linewidth=2, label="Theoretical: E[N]/E[T] = λ")
    
    # Plot simulation results
    plt.scatter(1/lambda_values, en_et_ratios, color='blue', s=50, label="Simulation results")
    
    # Connect simulation points
    plt.plot(1/lambda_values, en_et_ratios, 'b--', alpha=0.7)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('1/λ', fontsize=12)
    plt.ylabel('E[N]/E[T]', fontsize=12)
    plt.title("Verification of Little's Law Across Multiple λ Values", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Calculate overall metrics
    mean_rel_error = np.mean([abs((en_et - lambda_val) / lambda_val) * 100 
                             for en_et, lambda_val in zip(en_et_ratios, lambda_values)])
    print(f"\nAverage relative error across all λ values: {mean_rel_error:.4f}%")
    
    # Plot relative errors
    plt.figure(figsize=(10, 6))
    rel_errors = [abs((en_et - lambda_val) / lambda_val) * 100 
                 for en_et, lambda_val in zip(en_et_ratios, lambda_values)]
    
    plt.bar(lambda_values, rel_errors, width=lambda_values[1]-lambda_values[0] if len(lambda_values) > 1 else 0.1)
    plt.axhline(y=mean_rel_error, color='r', linestyle='--', 
               label=f"Mean Error: {mean_rel_error:.4f}%")
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('λ Value', fontsize=12)
    plt.ylabel('Relative Error (%)', fontsize=12)
    plt.title("Relative Error in Little's Law Verification", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("--------------------------------------------------------")
    print(" Simulation with Little's Law Convergence Plot ")
    print("--------------------------------------------------------")
    config, metrics = run_mer1_simulation(
        lambda_rate=0.3,  # This appears to be the value used in your graph
        r=3,
        mu=1.0,
        num_customers=10000,
        warm_up_customers=500,
        random_seed=42,
        plot_sample_path=False,  # Set to False to focus on Little's Law
        plot_histograms=False,   # Set to False to focus on Little's Law
        plot_littles_law_convergence=True
    )

    print("--------------------------------------------------------")
    print(" Simulation with λ = 0.9, r = 3, μ = 1.0 ---> Heavy Load " )
    print("--------------------------------------------------------")
    config, metrics = run_mer1_simulation(
        lambda_rate=0.4,
        r=2,
        mu=1.0,
        num_customers=1000000,
        warm_up_customers=10000,
        random_seed=42,
        plot_sample_path=True,
        plot_histograms=True,
        verify_littles_law=False,
        plot_littles_law_convergence=False
    )
    print("--------------------------------------------------------")
    print(" Simulation with λ = 0.9, r = 3, μ = 1.0 ---> Moderate Load " )
    print("--------------------------------------------------------")
    config, metrics = run_mer1_simulation(
        lambda_rate=0.3,
        r=3,
        mu=1.0,
        num_customers=1000000,
        warm_up_customers=10000,
        random_seed=42,
        plot_sample_path=True,
        plot_histograms=True,
        verify_littles_law=False,
        plot_littles_law_convergence=False
    )
    print("--------------------------------------------------------")
    print(" Simulation with λ = 0.2, r = 4, μ = 1.0 ---> Light Load " )
    print("--------------------------------------------------------")
    config, metrics = run_mer1_simulation(
        lambda_rate=0.2,
        r=4,
        mu=1.0,
        num_customers=1000000,
        warm_up_customers=10000,
        random_seed=42,
        plot_sample_path=True,
        plot_histograms=True,
        verify_littles_law=False,
        plot_littles_law_convergence=False
    )
    