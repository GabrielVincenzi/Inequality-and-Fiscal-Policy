import mesa
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from scipy.stats import beta
from customLoss import SmoothedParamLoss
from dqn import SimpleNN, DeepQNet

class SimpleAgent(mesa.Agent):
    def __init__(self, unique_id, wealth, environment):
        super().__init__(unique_id, environment)

        # Environmental variables
        self.unique_id = unique_id
        self.environment = environment
        self.simulate = self.environment.simulate
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
        self.delta = self.environment.delta
        self.beta = self.environment.beta
        self.gamma = self.environment.gamma
        self.psi = self.environment.psi
        self.v = self.environment.v
        self.aging = self.environment.aging
        self.gov = self.environment.gov
        self.horizon = self.environment.horizon
        if hasattr(self.environment, "scaler") and self.environment.scaler is not None:
            self.scaler = self.environment.scaler
        else:
            self.scaler = None
        
        # Agent variables
        self.wealth = wealth
        self.income = 0
        self.occupation = ''
        self.prob_supply = 1
        self.prob_demand = 1
        self.age_count = 15
        self.productivity = np.random.normal(1, 1)

        # Updating variables
        self.alpha_demand = 10
        self.alpha_supply = 10
        self.beta_demand = 2
        self.beta_supply = 2
        self.prob_demand = beta(self.alpha_demand, self.beta_demand)
        self.prob_supply = beta(self.alpha_supply, self.beta_supply)

        # Gov variables
        if self.gov and not self.taxSchedule:
            self.tau_income = self.environment.tau_income
            self.tau_wealth = self.environment.tau_wealth
        elif self.gov and self.taxSchedule:
            self.tau_income = 0
            self.tau_wealth = 1 - self.environment.lambda_wealth * (self.wealth)**(-self.environment.gamma_wealth)
        else:
            self.tau_income = 0
            self.tau_wealth = 0

        # RL variables
        self.hidden_steps = self.environment.hidden_steps
        self.thinking = self.environment.thinking
        self.state_dim = self.environment.state_dim
        self.action_dim = self.environment.action_dim
        self.epsilon = self.environment.epsilon

        # Network
        model_path = self.environment.model_path
        if os.path.exists(model_path):
            pass
        else:
            print("No pre-trained model found. An error occured in pre-training.")

        self.policy_network = DeepQNet(self.state_dim, self.action_dim, 256) if self.gov else SimpleNN(self.state_dim, self.action_dim, 256)
        self.policy_network.load_state_dict(torch.load(model_path))
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.step_count = 0

        self.frozen_state = self.update_frozen_state()

    
    def utility(self, income, labor_used):
        return self.delta * income - labor_used
    
    def returns(self, r0, r1, q):
        return r0 if np.random.binomial(1, q) else r1

    def get_state(self):
        enough_demand = 1 if self.environment.labor_demand >= 1 else 0
        enough_supply = 1 if self.environment.labor_supply >= self.mu else 0
        if self.gov:
            state_values = [
                self.wealth, self.tau_income, self.tau_wealth,
                self.delta, self.mu, self.interest_rate,
                self.investment_needed, self.r0_se, self.r1_se,
                self.r_mul, self.q_se, self.q_en, self.v,
                enough_demand, enough_supply
            ]
        else:
            state_values = [
                self.wealth,
                self.delta, self.mu, self.interest_rate,
                self.investment_needed, self.r0_se, self.r1_se,
                self.r_mul, self.q_se, self.q_en, self.v,
                enough_demand, enough_supply
            ]

        # Convert to torch tensor
        if self.scaler is not None:
            state_df = pd.DataFrame([state_values], columns=self.scaler.feature_names_in_)
            scaled_state = self.scaler.transform(state_df)
            return torch.tensor(scaled_state[0], dtype=torch.float32)
        else:
            return torch.tensor(state_values, dtype=torch.float32)
            
    def update_frozen_state(self):
        frozen_state = [self.wealth, self.alpha_demand, self.alpha_supply, self.beta_demand, self.beta_supply, self.v, self.psi, self.age_count]
        return frozen_state
    
    def update_from_frozen_state(self):
        self.wealth, self.alpha_demand, self.alpha_supply, self.beta_demand, self.beta_supply, self.v, self.psi, self.age_count = self.frozen_state
    
    def decay_epsilon(self, min_epsilon=0.001, decay_rate=0.995):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def action_outcomes(self, action, forward=False):
        if action == 0: # subsistence
            income = self.interest_rate * self.wealth
            reward = self.utility(income, 0)
        elif action == 1: # worker
            income = self.interest_rate * self.wealth + self.v * (1-self.tau_income) * 0.7
            reward = self.utility(income, 1)
        elif action == 2: # self-employed
            r = self.returns(self.r0_se, self.r1_se, self.q_se)
            income = self.wealth * self.interest_rate + (self.investment_needed * (r - self.interest_rate)) * (1-self.tau_income)
            reward = self.utility(income, 1)
        elif action == 3: # entrepreneur
            r = self.returns(self.r0_en, self.r1_en, self.q_en)
            income = self.wealth * self.interest_rate + (self.mu * self.investment_needed * (r - self.interest_rate) - self.mu * self.v * 0.5) * (1-self.tau_income)
            reward = self.utility(income, 1)
        else: 
            return ('Invalid action')
        
        if not forward:
            if (
                (action == 2 and self.wealth < self.investment_needed) or
                (action == 3 and self.wealth < self.investment_needed * self.mu* 0.5) or
                (action == 1 and self.environment.labor_demand < 1) or
                (action == 3 and self.environment.labor_supply < self.mu) or
                (self.wealth + income < 0 or income < 0)
            ):
                reward, income = -10, 0
        else:
            if (
                (action == 2 and self.wealth < self.investment_needed) or
                (action == 3 and self.wealth < self.investment_needed * self.mu* 0.5) or
                (self.wealth + income < 0 or income < 0)
            ):
                reward, income = -10, 0
            elif (action == 1 and self.environment.labor_demand < 1):
                prob = self.sample_demand()
                reward *= prob
                income *= prob
            elif (action == 3 and self.environment.labor_supply < self.mu):
                prob = self.sample_supply()
                reward *= prob
                income *= prob

        return torch.tensor(reward, dtype=torch.float32), income
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        with torch.no_grad():
            return torch.argmax(self.policy_network(state.unsqueeze(0))).item()
        
    def update_demand(self, x):
        """Update demand distribution with one observation x ∈ {0,1}"""
        self.alpha_demand += x
        self.beta_demand += 1 - x
        self.prob_demand = beta(self.alpha_demand, self.beta_demand)

    def update_supply(self, x):
        """Update supply distribution with one observation x ∈ {0,1}"""
        self.alpha_supply += x
        self.beta_supply += 1 - x
        self.prob_supply = beta(self.alpha_supply, self.beta_supply)

    def sample_demand(self):
        return self.prob_demand.rvs()

    def sample_supply(self):
        return self.prob_supply.rvs()
    
        
    def get_occupation(self, action, reward):
        occupation_map = {
            0: "subsistence",
            1: "working",
            2: "self-employment",
            3: "entrepreneurship",
        }

        if self.environment.labor_demand >= 0:
            x = 1
        elif self.environment.labor_demand < 0:
            x = 0
        self.update_demand(x)

        if self.environment.labor_supply >= self.mu:
            x = 1
        elif self.environment.labor_supply < self.mu:
            x = 0
        self.update_supply(x)
        
        #if self.environment.labor_demand >= 0:
        #    self.prob_demand = min(1, self.prob_demand * 1.2)
        #elif self.environment.labor_demand < 0:
        #    self.prob_demand *= 0.9
        #elif self.environment.labor_supply >= self.mu:
        #    self.prob_supply = min(1, self.prob_supply * 1.2)
        #elif self.environment.labor_supply < self.mu:
        #    self.prob_supply *= 0.9

        if action == 1 and reward >= 0:
            self.environment.labor_demand -= 1
        elif action == 3 and reward >= 0:
            self.environment.labor_supply -= self.mu
        
        return occupation_map.get(action, " ")
    
    def future_rewards(self):
        initial_wealth = self.wealth
        actions = list(range(self.action_dim))
        horizon = self.horizon
        rewards = np.zeros([len(actions), horizon + 1])
        best_rewards_per_action = np.full(len(actions), -np.inf)
        action_sequences = np.array(list(itertools.product(actions, repeat=horizon)))

        for sequence in action_sequences:
            first_action = sequence[0]
            w = initial_wealth
            rewards = []

            for h, a in enumerate(sequence):
                if h == 0:
                    reward, income = self.action_outcomes(a, forward=False)
                else:
                    reward, income = self.action_outcomes(a, forward=True)
                w = (1-self.tau_wealth) * w + income * (self.gamma)
                rewards.append((self.beta ** h) * reward)

            total_reward = sum(rewards)

            if total_reward > best_rewards_per_action[first_action]:
                best_rewards_per_action[first_action] = total_reward

        best_rewards_tensor = torch.tensor(best_rewards_per_action, dtype=torch.float32)
        #print(f'Best Rewards: {best_rewards_tensor}')
        return best_rewards_tensor
        
    def train(self, state):
        #print(f"State: {state}")
        target_rewards = self.future_rewards()
        predicted_rewards = self.policy_network(state.unsqueeze(0)).squeeze(0)
        #print(f"Target rewards: {target_rewards}")
        #print(f"Predicted rewards: {predicted_rewards}")
        
        loss_fn = SmoothedParamLoss()
        loss = loss_fn(predicted_rewards, target_rewards, self.step_count)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def simulate_path(self):
        for step in range(self.thinking if self.step_count == 0 else self.hidden_steps):
            state = self.get_state()
            action = self.choose_action(state)
            reward, income = self.action_outcomes(action)
            self.train(state)

            #if not self.simulate:
                #print(f"Step {self.step_count} for agent {self.unique_id} completed:")
                #print(f'Action: {action} with reward: {reward} starting from wealth: {self.wealth}')
                #print(f"Frozen State updated to: {self.frozen_state}")
                #print(f"Supply: {self.environment.labor_supply} and Demand: {self.environment.labor_demand}")

        # Update states
        self.wealth = max((1-self.tau_wealth) * self.wealth + income* (self.gamma), 0)
        self.occupation = self.get_occupation(action, reward)
        self.income = income
        
        return reward, income


    def update_macrovars(self):
        self.psi = self.environment.psi
        self.v = self.environment.v
        if self.gov and not self.taxSchedule:
            self.tau_income = self.environment.tau_income
            self.tau_wealth = self.environment.tau_wealth
        elif self.gov and self.taxSchedule:
            self.tau_income = 1 - self.environment.lambda_income * (self.income)**(-self.environment.gamma_income) if self.income > 0 else 0
            self.tau_wealth = 1 - self.environment.lambda_wealth * (self.wealth)**(-self.environment.gamma_wealth) if self.wealth > 0 else 0

    
    def age(self):
        if self.age_count >= 45:
            self.wealth = (1-self.gamma) * self.wealth
            #self.policy_network.load_state_dict(torch.load(self.environment.model_path))
            self.age_count = 15

        self.age_count += 1

    
    def step(self):
        self.simulate = self.environment.simulate
        if not self.simulate and self.gov: self.update_from_frozen_state()
        self.update_macrovars()
        self.simulate_path()
        if not self.simulate: self.decay_epsilon()

        if self.aging:
            self.age()
            
        self.step_count += 1

        if not self.simulate and self.gov: self.frozen_state = self.update_frozen_state()