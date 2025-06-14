import mesa
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import lognorm, pareto
import joblib
import random

from simpleAgent import SimpleAgent
from government import Government
from pretrain import pretrain_model
from pretrain_wg import pretrain_model_withgov

np.random.seed(54)


class Environment(mesa.Model):
    """
    Environment:
    - Sets environmental variables.
    - Simulates interactions of individuals.
    - Collects and stores data.
    """

    def __init__(self, N, think=5, aging=True, gov=False, taxSchedule=False):
        super().__init__()

        # Mesa variables
        self.num_agents = N
        self.thinking = think
        self.aging = aging
        self.gov = gov
        self.taxSchedule = taxSchedule
        self.schedule = mesa.time.BaseScheduler(self)
        self.horizon = 3 if self.gov else 3

        # Environmental variables
        self.investment_needed = 500
        self.interest_rate = 0.03
        self.simulate = False
        self.tax_records = []

        # Returns for self-employed and enterpreneurs
        self.r0_se = 0.10
        self.r1_se = 0.20
        self.r_mul = 0.0
        self.r0_en = (1 - self.r_mul) * self.r0_se
        self.r1_en = (1 + self.r_mul) * self.r1_se
        self.q_se = 0.2
        self.q_en = (self.r0_se * self.q_se - self.r1_se * (self.r_mul + self.q_se)) / ((1 - self.r_mul) * self.r0_se - (1 + self.r_mul) * self.r1_se)
        self.r_exp = self.r0_se * self.q_se + self.r1_se * (1 - self.q_se)
        
        self.psi = 0.1
        self.mu = 10
        self.beta = 0.9
        self.gamma = 0.8
        self.delta = self.gamma**(self.gamma) * (1-self.gamma)**(1-self.gamma)

        # Gov variables
        if self.gov:
            self.tau_income = 0
            self.tau_wealth = 0

        # Progressive Tax schedule
        if self.taxSchedule:
            self.lambda_income = 1
            self.lambda_wealth = 1
            self.gamma_income = 0
            self.gamma_wealth = 0

        # RL variables
        self.state_dim = 15 if self.gov else 13  # The number of features representing the state
        self.action_dim = 4  # Number of possible actions
        self.hidden_steps = 1
        self.epsilon = 0.1

        # Wage variables
        self.labor_demand = 0
        self.labor_supply = 0
        self.v_min = 1/self.delta
        self.v_max = ((self.mu - 1) / self.mu) * self.investment_needed * (self.r_exp - self.interest_rate)
        self.v = self.v_min + (self.v_max - self.v_min) * self.psi / (self.psi + 1)

        self.frozen_environment = self.update_frozen_environment()

        # Model static inputs: delta, interest_rate, investment_needed, r0_se, r1_se, r_mul, q_se, q_en, mu,
        # Model changing inputs: wealth, v,  enough_en, enough_wr 
        self.model_path = (f"{'model_paths_withGov' if gov else 'model_paths'}/"
            f"d{str(round(self.delta, 2)).replace('.', '_')}"
            f"r{str(self.interest_rate).replace('.', '_')}"
            f"I{str(self.investment_needed).replace('.', '_')}"
            f"r0se{str(self.r0_se).replace('.', '_')}"
            f"r1se{str(self.r1_se).replace('.', '_')}"
            f"rm{str(self.r_mul).replace('.', '_')}"
            f"qse{str(self.q_se).replace('.', '_')}"
            f"qen{str(round(self.q_en, 2)).replace('.', '_')}"
            f"mu{str(self.mu).replace('.', '_')}.pth"
        )

        if os.path.exists(self.model_path):
            print(f"Loading pre-trained model from {self.model_path}")
        else:
            print("No pre-trained model found. Initializing a new model.")
            if gov:
                pretrain_model_withgov(self.delta, self.interest_rate, self.investment_needed, self.r0_se, self.r1_se, self.r_mul, self.q_se, self.q_en, self.mu, self.v_min, self.v_max)
            else:
                pretrain_model(self.delta, self.interest_rate, self.investment_needed, self.r0_se, self.r1_se, self.r_mul, self.q_se, self.q_en, self.mu, self.v_min, self.v_max)

        if self.gov: self.scaler = joblib.load(f"state_scaler_{self.mu}.pkl")

        # Generate lognormal wealth distribution
        #mean, sigma = 6, 1.2
        #scale = np.exp(mean)
        #wealth_values = lognorm.rvs(sigma, scale=scale, size=self.num_agents, random_state=54)

        # Generate a lognormal + Pareto distribution
        #theta = 940
        #mu = 3.20
        #sigma = 1.92
        #alpha = 1.01
        #wealth_values = self.sample_composite_lognorm_pareto(theta, mu, sigma, alpha, size=self.num_agents)
        wealth_values = [10000] * int(1*self.num_agents/(self.mu+1)) + [1] * (self.num_agents - int(1*self.num_agents/(self.mu+1)))
        wealth_values.sort()
        self.starting_wealth_values = wealth_values

        # Create agents
        for i in range(self.num_agents):
            agent = SimpleAgent(i, wealth_values[i], self)
            self.schedule.add(agent)

        # Create government
        if self.gov:
            self.government = Government(self)

        # Data collector to track wealth and consumption distribution
        self.datacollector = mesa.datacollection.DataCollector(
            agent_reporters={
                "Wealth": lambda a: a.wealth,
                "Occupation": lambda a: a.occupation,
            },
            model_reporters={
                "Wage": lambda m: m.v,
            }
        )

    def sample_composite_lognorm_pareto(self, theta, mu, sigma, alpha, size=1):
        """Probability mass divided at the threshold theta between a lognormal and a Pareto area."""
        p = lognorm.cdf(theta, s=sigma, scale=np.exp(mu))
        
        samples = np.empty(size)
        u = np.random.uniform(0, 1, size)
        
        # Number of samples from each component
        n_lognorm = np.sum(u <= p)
        n_pareto = size - n_lognorm
        
        # Sample from truncated lognormal (accept-reject)
        if n_lognorm > 0:
            lognorm_samples = []
            while len(lognorm_samples) < n_lognorm:
                candidate = lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_lognorm*2)
                accepted = candidate[candidate <= theta]
                lognorm_samples.extend(accepted.tolist())
            lognorm_samples = np.array(lognorm_samples[:n_lognorm])
            samples[u <= p] = lognorm_samples
        
        # Sample from Pareto tail
        if n_pareto > 0:
            # Pareto in scipy: pdf ~ (scale/x)^(alpha+1), support x >= scale
            pareto_samples = pareto.rvs(b=alpha, scale=theta, size=n_pareto)
            samples[u > p] = pareto_samples
        
        return samples

    def update_frozen_environment(self):
        frozen_env = [self.labor_demand, self.labor_supply, self.v, self.psi]
        return frozen_env
    
    def update_from_frozen_environment(self):
        self.labor_demand, self.labor_supply, self.v, self.psi = self.frozen_environment

    def calculate_macrovars(self):
        if not self.simulate and self.gov: self.update_from_frozen_environment()
        occupations = np.array([a.occupation for a in self.schedule.agents])
        self.labor_supply = np.sum(occupations == "working")  
        self.labor_demand = np.sum(occupations == "entrepreneurship") * self.mu 
        self.psi = (
            self.labor_demand / self.labor_supply 
            if self.labor_supply != 0 
            else 0
        )
        self.v_min = 1/(self.delta*(1-self.tau_income)) if self.gov else 1/self.delta
        self.v = self.v_min + (self.v_max - self.v_min) * self.psi / (self.psi + 1)
        if not self.simulate and self.gov: self.frozen_environment = self.update_frozen_environment()


    def step(self):
        """Advance the model by one step and collects the data at the end of the step."""
        if not self.simulate and self.gov and not self.taxSchedule:
            print("---------- Government simulation ----------")
            self.tau_income, self.tau_wealth = self.government.step()
            self.tax_records.append([self.tau_income, self.tau_wealth])
            self.simulate = False
            print("---------- Environment simulation ----------")
        elif not self.simulate and self.gov and self.taxSchedule:
            print("---------- Government simulation TS ----------")
            self.lambda_income, self.lambda_wealth, self.gamma_income, self.gamma_wealth = self.government.step()
            self.tax_records.append([self.lambda_income, self.lambda_wealth, self.gamma_income, self.gamma_wealth])
            self.simulate = False
            print("---------- Environment simulation TS ----------")
        self.schedule.step()
        self.calculate_macrovars()
        self.datacollector.collect(self)
        if self.gov and not self.taxSchedule: print(f"Government choose Income Tax: {self.tau_income*100:.2f} and Wealth tax. {self.tau_wealth*100:.2f}")
        if self.gov and self.taxSchedule: print(f"Government choose Income Tax:{self.lambda_income:.4f}, {self.gamma_income:.4f} and Wealth tax. {self.lambda_wealth:.4f}, {self.gamma_wealth:.4f}")


    def run(self, steps, tau_income=None, tau_wealth=None, lambda_income=None, lambda_wealth=None, gamma_income=None, gamma_wealth=None, simulate=False):
        """Run the model for given steps."""
        self.num_steps = steps
        self.simulate = simulate
        if self.simulate and not self.taxSchedule: 
            self.tau_income, self.tau_wealth = tau_income, tau_wealth
        if self.simulate and self.taxSchedule: 
            self.lambda_income, self.lambda_wealth, self.gamma_income, self.gamma_wealth = lambda_income, lambda_wealth, gamma_income, gamma_wealth
        for i in range(self.num_steps):
            self.step()


    def get_data(self):
        """Get a dataframe comprising all informations of the model."""
        df = self.datacollector.get_agent_vars_dataframe()
        return df
    

    def gini(self, x, w=None):
        """Calculate Gini coefficient using the provided formula."""
        x = np.asarray(x)
        if w is not None:
            w = np.asarray(w)
            sorted_indices = np.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_w = w[sorted_indices]
            cumw = np.cumsum(sorted_w, dtype=float)
            cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
            return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                    (cumxw[-1] * cumw[-1]))
        else:
            sorted_x = np.sort(x)
            n = len(x)
            cumx = np.cumsum(sorted_x, dtype=float)
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    

    def plot(self):
        """
        Plots four key visualizations in a 3x2 layout:
        - Top-left: Occupation Shares Over Time
        - Top-right: Wealth Distribution
        - Bottom-left: Occupation Over Time
        - Bottom-right: Wealth Stream
        """
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        
        # Fetch data
        data = self.get_data()
        if data is None:
            print("Warning: No data available.")
            return
        
        occupation_map = {
            'subsistence': 0,
            'working': 1,
            'self-employment': 2,
            'entrepreneurship': 3
        }
        
        cmap = cm.get_cmap('PuOr', len(occupation_map))
        occupation_colors = {occ: cmap(i / (len(occupation_map) - 1)) for occ, i in zip(occupation_map.keys(), range(len(occupation_map)))}

    
        # ---- Top-left: Occupation Shares Over Time ----
        if "Occupation" in data.columns:
            occupation_counts = data.groupby(level="Step")['Occupation'].value_counts().unstack(fill_value=0)
            occupation_shares = occupation_counts.div(occupation_counts.sum(axis=1), axis=0)

            if self.gov: occupation_shares = occupation_shares.iloc[::6].reset_index(drop=True)

            axes[0, 0].stackplot(
                occupation_shares.index,
                [
                    occupation_shares[occ] if occ in occupation_shares else pd.Series(0, index=occupation_shares.index)
                    for occ in occupation_map.keys()
                ],
                colors=[occupation_colors[occ] for occ in occupation_map.keys()],
                labels=[occ.capitalize() for occ in occupation_map.keys()]
            )

            axes[0, 0].set_title("Occupation Shares Over Time", fontsize=18)
            axes[0, 0].set_xlabel("Time Step", fontsize=14)
            axes[0, 0].set_ylabel("Share of Agents", fontsize=14)
            
        
        # ---- Top-right: Wealth Distribution ----
        variable_name = "wealth"
        final_values = [getattr(agent, variable_name, None) for agent in self.schedule.agents]
        occupations = [agent.occupation for agent in self.schedule.agents]

        sigma, loc, scale = lognorm.fit(final_values, floc=0)
        print(f"Sigma: {sigma}, Mean: {np.log(scale)}, Scale: {scale}.")
        
        if final_values:
            bins = np.linspace(min(final_values), max(final_values), int(self.num_agents / 2))
            occupation_values = {}
            for value, occupation in zip(final_values, occupations):
                if occupation in occupation_map:
                    occupation_values.setdefault(occupation, []).append(value)

            bottom_array = np.zeros(len(bins) - 1)
            for occupation, values in occupation_values.items():
                if not values:
                    continue
                hist_values, _ = np.histogram(values, bins=bins)
                axes[0, 1].bar(bins[:-1], hist_values, width=np.diff(bins), bottom=bottom_array, color=occupation_colors[occupation], label=occupation.capitalize())
                bottom_array += hist_values

            axes[0, 1].set_title(f"Stacked Distribution of {variable_name.capitalize()} by Occupation", fontsize=18)
            axes[0, 1].set_xlabel(variable_name.capitalize(), fontsize=14)
            axes[0, 1].set_ylabel("Number of Agents", fontsize=14)
            axes[0, 1].legend(title="Occupation", loc='upper right')


        # ---- Middle-right: Gini over time ----
        if variable_name.capitalize() in data.columns:

            # Initialize list to store Gini coefficients for each time step
            gini_over_time = []
            
            # Get all unique time steps (years or steps in the data)
            unique_steps = sorted(data.index.get_level_values("Step").unique())
            selected_steps = unique_steps[::6]
            plot_steps = range(1, len(selected_steps) + 1)

            for step in (selected_steps if self.gov else unique_steps):
                # Extract wealth values at each time step
                wealth_values = data.loc[data.index.get_level_values("Step") == step, variable_name.capitalize()]
                
                # Calculate Gini coefficient for this time step
                gini_value = self.gini(wealth_values.values)
                
                # Append Gini coefficient for the current time step
                gini_over_time.append(gini_value)

            # Plot the Gini coefficient over time
            if self.gov: axes[1, 1].plot(plot_steps, gini_over_time, color='red', linewidth=2, label="Gini Coefficient")
            else: axes[1, 1].plot(unique_steps, gini_over_time, color='red', linewidth=2, label="Gini Coefficient")

            # Set labels and title
            axes[1, 1].set_title("Gini Coefficient Over Time", fontsize=18)
            axes[1, 1].set_xlabel("Time Step", fontsize=14)
            axes[1, 1].set_ylabel("Gini Coefficient", fontsize=14)
            axes[1, 1].grid(True)
            axes[1, 1].legend()


        # ---- Middle-left: Taxation ----
        if self.gov and not self.taxSchedule:
            tau_income_vals, tau_wealth_vals = zip(*self.tax_records)

            axes[1, 0].plot(tau_income_vals, label='Income Tax Rate')
            axes[1, 0].plot(tau_wealth_vals, label='Wealth Tax Rate')
            axes[1, 0].set_xlabel('Time Step', fontsize=14)
            axes[1, 0].set_ylabel('Tax Rate', fontsize=14)
            axes[1, 0].set_title('Income and Wealth Tax Rates Over Time', fontsize=18)
            axes[1, 0].grid(True)
            axes[1, 0].legend()

        if self.gov and self.taxSchedule:
            progressivity_df = pd.DataFrame(self.tax_records)
            progressivity_df.to_csv('progressivity_records.csv', index=False, header=False)
            lambda_income_vals, lambda_wealth_vals, gamma_income_vals, gamma_wealth_vals = zip(*self.tax_records)

            axes[1, 0].plot(lambda_income_vals, label='Lambda Income')
            axes[1, 0].plot(lambda_wealth_vals, label='Lambda Wealth')
            axes[1, 0].plot(gamma_income_vals, label='Gamma Income')
            axes[1, 0].plot(gamma_wealth_vals, label='Gamma Wealth')
            axes[1, 0].set_xlabel('Time Step', fontsize=14)
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_ylabel('Tax Rate', fontsize=14)
            axes[1, 0].set_title('Income and Wealth Tax Rates Over Time', fontsize=18)
            axes[1, 0].grid(True)
            axes[1, 0].legend()
            

        # ---- Bottom-left: Occupation Over Time ----
        data['Occupation_numeric'] = data['Occupation'].map(occupation_map)
        df_pivot = data.pivot_table(index='Step', columns='AgentID', values='Occupation_numeric', aggfunc='first')
        df_pivot_sub = df_pivot.iloc[::6].reset_index(drop=True)
        plot_steps = range(1, len(df_pivot_sub.index) + 1)
        
        for agent_id in df_pivot.columns:
            occupation = data.loc[data.index.get_level_values('AgentID') == agent_id, 'Occupation'].iloc[0]
            if self.gov: axes[2, 0].plot(plot_steps, df_pivot_sub[agent_id], alpha=0.3, color='black')
            else: axes[2, 0].plot(df_pivot.index, df_pivot[agent_id], alpha=0.3, color='black')
        
        axes[2, 0].set_title("Agent Occupation Over Time", fontsize=18)
        axes[2, 0].set_xlabel("Time Step", fontsize=14)
        axes[2, 0].set_ylabel("Occupation", fontsize=14)
        axes[2, 0].set_yticks([0, 1, 2, 3])
        axes[2, 0].set_yticklabels(['Subsistence', 'Working', 'Self-employment', 'Entrepreneurship'])
        
        # ---- Bottom-right: Wealth Stream ----
        if variable_name.capitalize() in data.columns:

            for agent_id, agent_data in data[variable_name.capitalize()].groupby(level="AgentID"):
                agent_data_sub = agent_data.iloc[::6]
                plot_steps = range(1, len(agent_data_sub) + 1)
                #occupation = data.loc[data.index.get_level_values('AgentID') == agent_id, 'Occupation'].iloc[0]
                if self.gov: axes[2, 1].plot(plot_steps, agent_data_sub.values, alpha=0.5, color='black')
                else: axes[2, 1].plot(agent_data.index.get_level_values("Step"), agent_data.values, alpha=0.5, color='black')

            # Set labels for the primary y-axis (wealth)
            axes[2, 1].set_title(f"{variable_name.capitalize()} Paths & Wage Over Time", fontsize=18)
            axes[2, 1].set_xlabel("Time Step", fontsize=14)
            axes[2, 1].set_ylabel(variable_name.capitalize(), fontsize=14)
            axes[2, 1].grid(False)

            # Create a secondary y-axis for wage v
            ax2 = axes[2, 1].twinx()

            # Retrieve model-level data (aggregates) for wage v
            model_data = self.datacollector.get_model_vars_dataframe()
            time_steps = model_data.index[::6]
            plot_steps = range(1, len(time_steps) + 1)

            # Plot wage v on the secondary y-axis
            if self.gov: ax2.plot(plot_steps, model_data["Wage"].iloc[::6], color='blue', linewidth=2, label="Wage")
            else: ax2.plot(model_data.index, model_data["Wage"], color='blue', linewidth=2, label="Wage")
            ax2.set_ylabel("Wage", fontsize=14, color='blue')

            # Add a legend for wage v
            ax2.legend(loc="upper left")
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()
    

