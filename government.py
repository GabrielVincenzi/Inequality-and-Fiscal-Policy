import mesa
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import RobustNN

class Government(mesa.Agent):
    def __init__(self, environment):
        super().__init__('Gov', environment)

        # Environmental variables
        self.environment = environment
        self.taxSchedule = self.environment.taxSchedule
        self.investment_needed = self.environment.investment_needed
        self.interest_rate = self.environment.interest_rate
        self.r0_se = self.environment.r0_se
        self.r1_se = self.environment.r1_se
        self.r0_en = self.environment.r0_en
        self.r1_en = self.environment.r1_en
        self.r_mul = self.environment.r_mul
        self.q_se = self.environment.q_se
        self.q_en = self.environment.q_en
        self.mu = self.environment.mu
        self.psi = self.environment.psi
        self.v = self.environment.v
        self.r_exp = self.r0_se * self.q_se + self.r1_se * (1 - self.q_se)
        
        # Government variables (expects to have same firms as present)
        self.horizon = 5
        self.state_dim = 13 if not self.taxSchedule else 14
        self.output_dim = 2 if not self.taxSchedule else 4
        self.tau_income = 0
        self.tau_wealth = 0
        self.gini_index = 0.65

        # RL variables
        self.hidden_steps = self.environment.hidden_steps
        self.epsilon = self.environment.epsilon

        # Network
        self.model_path = (f"{'model_paths_gov' if not self.taxSchedule else 'model_paths_gov-ts'}/"
            f"r{str(self.interest_rate).replace('.', '_')}"
            f"I{str(self.investment_needed).replace('.', '_')}"
            f"r0se{str(self.r0_se).replace('.', '_')}"
            f"r1se{str(self.r1_se).replace('.', '_')}"
            f"rm{str(self.r_mul).replace('.', '_')}"
            f"qse{str(self.q_se).replace('.', '_')}"
            f"qen{str(round(self.q_en, 2)).replace('.', '_')}"
            f"mu{str(self.mu).replace('.', '_')}.pth"
        )

        self.policy_network = RobustNN(self.state_dim, self.output_dim, 256)
        if os.path.exists(self.model_path):
            self.policy_network.load_state_dict(torch.load(self.model_path))
        else:
            print("No pre-trained model found. Initializing a new model.")
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01, weight_decay=1e-4)
        self.step_count = 0

    def get_state(self):
        combined_states = [
            torch.tensor(self.mu, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(self.interest_rate, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(self.investment_needed, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(self.r0_se, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(self.r1_se, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(self.r_mul, dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.q_se, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(self.q_en, dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.v, dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.n_su, dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.n_w, dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.n_se, dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.n_en, dtype=torch.float32).unsqueeze(0)
        ]

        if self.taxSchedule:
            combined_states.append(torch.tensor(self.gini_index, dtype=torch.float32).unsqueeze(0))

        combined_state = torch.cat(combined_states, dim=-1)
        return combined_state
    
    def decay_epsilon(self, min_epsilon=0.001, decay_rate=0.995):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    
    def simulate_environment(self, tau_income, tau_wealth):
        self.environment.run(self.horizon, tau_income, tau_wealth, simulate=True)
        final_wealth = sum(agent.wealth for agent in self.environment.schedule.agents)
        wealth_by_occupation = {
            "subsistence": 0.0,
            "working": 0.0,
            "self-employment": 0.0,
            "entrepreneurship": 0.0
        }
        for agent in self.environment.schedule.agents:
            wealth_by_occupation[agent.occupation] += agent.wealth

        return final_wealth, wealth_by_occupation
    
        
    def train(self, state):
        """Divides in self.tax for the backpropagation and tax (detatched) for the simulations."""
        taxes = self.policy_network(state.unsqueeze(0)).squeeze(0)
        tau_income, tau_wealth = taxes[0], taxes[1]
        print(f"Taxes: {taxes}")

        taxes_for_sim = taxes.detach().cpu().numpy()
        tau_income_val, tau_wealth_val = taxes_for_sim[0], taxes_for_sim[1]
        self.tau_income, self.tau_wealth = tau_income_val, tau_wealth_val
        agg_wealth, wbo = self.simulate_environment(tau_income_val, tau_wealth_val)

        agg_income = ((wbo["working"] * self.v) +
                          (wbo["self-employment"] * self.investment_needed * (self.r_exp - self.interest_rate)) +
                          (wbo["entrepreneurship"] * (self.mu * self.investment_needed * (self.r_exp - self.interest_rate) - self.mu * self.v))
                          )

        expenditure_rate = 0.01
        p = 0.9
        loss = - p * agg_wealth + (1-p) * abs(expenditure_rate * agg_wealth - (agg_wealth * tau_wealth + agg_income * tau_income))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()


    def trainSchedule(self, state):
        """
        Train a government that decides the level of progressivity instead of a flat tax with 1 - lambda * x ** (-gamma).
        - If gamma = 1 : full progressivity
        - If gamma = 0 : constant tax rate
        """
        output = self.policy_network(state.unsqueeze(0)).squeeze(0)
        lambda_income, lambda_wealth, gamma_income, gamma_wealth = output

        output_for_sim = output.detach().cpu().numpy()
        lambda_income_val, lambda_wealth_val, gamma_income_val, gamma_wealth_val = output_for_sim
        self.lambda_income, self.lambda_wealth, self.gamma_income, self.gamma_wealth = lambda_income_val, lambda_wealth_val, gamma_income_val, gamma_wealth_val

        self.environment.run(self.horizon, lambda_income=lambda_income_val, lambda_wealth=lambda_wealth_val, gamma_income=gamma_income_val, gamma_wealth=gamma_wealth_val, simulate=True)

        agg_wealth_revenues = 0
        agg_income_revenues = 0
        for agent in self.environment.schedule.agents:
            wealth = agent.wealth
            if agent.occupation == "subsistence":
                income = 0
            elif agent.occupation == "working":
                income = self.v
            elif agent.occupation == "self-employment":
                income = self.investment_needed * (self.r_exp - self.interest_rate)
            elif agent.occupation == "entrepreneurship":
                income = (self.mu * self.investment_needed * (self.r_exp - self.interest_rate) - self.mu * self.v)
            else:
                print('An agent had invalid occupation')

            wealth_tax = 1 - lambda_wealth * (1/wealth)**(gamma_wealth) if wealth != 0 else 0
            income_tax = 1 - lambda_income * (1/income)**(gamma_income) if income != 0 else 0

            agg_wealth_revenues += wealth * wealth_tax
            agg_income_revenues += income * income_tax

        agg_wealth = sum(agent.wealth for agent in self.environment.schedule.agents)

        expenditure_rate = 0.01
        p = 0.9
        abs_diff = abs(expenditure_rate * agg_wealth - (agg_wealth_revenues + agg_income_revenues))
        loss = - p * (agg_wealth) + (1-p) * abs_diff
        print(f"Agg wealth: {agg_wealth}, Abs diff: {abs_diff}, Loss: {loss}")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        

    
    def simulate_path(self):
        state = self.get_state()
        self.train(state) if not self.taxSchedule else self.trainSchedule(state)

        if not self.taxSchedule: print(f"-- Step {self.step_count} for Government with Income tax {self.tau_income} and Wealth tax {self.tau_wealth} --")


    def update_state(self):
        wealth_vals = np.array([a.wealth for a in self.environment.schedule.agents])
        self.gini_index = self.gini(wealth_vals)
        occupations = np.array([a.occupation for a in self.environment.schedule.agents])
        self.n_su = np.sum(occupations == "subsistence")
        self.n_w = np.sum(occupations == "working")  
        self.n_se = np.sum(occupations == "self-employment")  
        self.n_en = np.sum(occupations == "entrepreneurship")
        self.psi = self.environment.psi
        self.v = self.environment.v

    def get_taxrates(self):
        return self.tau_income, self.tau_wealth
    
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

    
    def step(self):
        self.update_state()
        self.simulate_path()
        self.decay_epsilon()
        self.step_count += 1

        #if self.step_count == self.environment.num_steps:
        torch.save(self.policy_network.state_dict(), self.model_path)

        if not self.taxSchedule:
            return self.tau_income, self.tau_wealth
        elif self.taxSchedule:
            return self.lambda_income, self.lambda_wealth, self.gamma_income, self.gamma_wealth