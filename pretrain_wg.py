import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from customDataset import TrainingDataset
from torch.optim import lr_scheduler
from dqn import SimpleNN, DeepQNet
from sklearn.preprocessing import StandardScaler
import joblib

import numpy as np
import pandas as pd
import itertools
import os


### Utility functions  
def generate_sequence(a_min, a_max, a_num, avecpar=0.30):
    # Generate a sequence using a nonlinear progression (exponential/logarithmic)
    lin = np.linspace(0.0, 1.0, a_num)
    sequence = a_min + (a_max - a_min) * (lin ** (1.0 / avecpar))
    return sequence

def utility(income, labor_used, delta):
        return delta * income - labor_used
    
def returns(r0, r1, q):
    random_draws = np.random.uniform(0, 1, size=1)
    return np.where(random_draws < q, r0, r1)

def future_rewards(state, action_dim, horizon, beta, action_outcomes_fn):
    num_states = len(state)
    best_rewards = np.full((num_states, action_dim), -np.inf)  # Initialize with -inf
    actions = list(range(action_dim))
    
    # Generate all possible action sequences for the given horizon
    action_sequences = np.array(list(itertools.product(actions, repeat=horizon)))

    for i, row in state.iterrows():
        initial_wealth = row["wealth"]  # Extract wealth for the state
        
        for sequence in action_sequences:
            first_action = sequence[0]  # Track first action in sequence
            w = initial_wealth
            rewards = []

            for h, a in enumerate(sequence):
                if h == 0:
                    reward, income = action_outcomes_fn(row, a, forward=False)
                else:
                    reward, income = action_outcomes_fn(row, a, forward=True)
                w = (1-row["tau_wealth"]) * w + income
                rewards.append((beta ** h) * reward)

            total_reward = sum(rewards)

            # Store best possible reward for the first action
            if total_reward > best_rewards[i, first_action]:
                best_rewards[i, first_action] = total_reward

    return best_rewards

def action_outcomes(state, action, forward=False):
        reward = pd.Series(np.nan, index=state.index)

        r0_en = (1- state["r_mul"]) * state["r0_se"]
        r1_en = (1 + state["r_mul"]) * state["r0_se"]

        if action == 0:
            income = state["interest_rate"] * state["wealth"]
            reward = utility(income, 0, state["delta"])
        elif action == 1:
            income = (state["interest_rate"] * state["wealth"] + state["v"] * (1-state["tau_income"]))
            reward = utility(income, 1, state["delta"])
        elif action == 2:
            r = returns(state["r0_se"], state["r1_se"], state["q_se"])
            income = state["wealth"]  * state["interest_rate"] + (state["investment_needed"] * (r - state["interest_rate"])) * (1-state["tau_income"])
            reward = utility(income, 1, state["delta"])
        elif action == 3: 
            r = returns(r0_en, r1_en, state["q_en"])
            income = state["wealth"] * state["interest_rate"] + (state["mu"] * state["investment_needed"] * (r - state["interest_rate"]) - state["mu"] * state["v"]) * (1-state["tau_income"])
            reward = utility(income, 1, state["delta"])
        else: 
            raise ValueError("Invalid action: Must be 0, 1, 2, or 3")
        
        # First invalid condition check for `forward == False`
        if not forward:
            if (
                (action == 2 and state["wealth"] < state["investment_needed"]) or
                (action == 3 and state["wealth"] < state["investment_needed"] * state["mu"]) or
                (action == 1 and state["enough_en"] == 0) or
                (action == 3 and state["enough_wr"] == 0) or
                (state["wealth"] + income < 0 or income < 0)
            ):
                return -10, 0  # Assign punishment reward

        # Second invalid condition check for `forward == True`
        else:
            if (
                (action == 2 and state["wealth"] < state["investment_needed"]) or
                (action == 3 and state["wealth"] < state["investment_needed"] * state["mu"]) or
                (state["wealth"] + income < 0 or income < 0)
            ):
                return -10, 0
            elif (
                (action == 1 and state["enough_en"] == 0) or
                (action == 3 and state["enough_wr"] == 0)
                ):
                reward *= 0.5
                income *= 0.5
        
        return reward, income
        


def pretrain_model_withgov(delta, interest_rate, investment_needed, r0_se, r1_se, r_mul, q_se, q_en, mu, v_min, v_max):
    model_name = (
        f"d{str(round(delta, 2)).replace('.', '_')}"
        f"r{str(interest_rate).replace('.', '_')}"
        f"I{str(investment_needed).replace('.', '_')}"
        f"r0se{str(r0_se).replace('.', '_')}"
        f"r1se{str(r1_se).replace('.', '_')}"
        f"rm{str(r_mul).replace('.', '_')}"
        f"qse{str(q_se).replace('.', '_')}"
        f"qen{str(round(q_en, 2)).replace('.', '_')}"
        f"mu{str(mu).replace('.', '_')}.pth"
    )

    state_columns = ["wealth", "tau_income", "tau_wealth", "delta", "mu", "interest_rate", "investment_needed",
                        "r0_se", "r1_se", "r_mul", "q_se", "q_en", "v", "enough_en", "enough_wr"]
    reward_columns = ['action_0', 'action_1', 'action_2', 'action_3']

    if os.path.exists("training_table"):
        print(f"Loading Training Table")
        training_table = pd.read_csv("training_table", sep=";")
    else:
        psi_ratio = np.linspace(0, 1, 20)
        v_values = v_min + (v_max - v_min) * psi_ratio
        wealth_values = generate_sequence(0, 7000, 50, avecpar=0.50)
        tau_income_values = np.linspace(0, 0.8, 9)
        tau_wealth_values = np.linspace(0, 0.8, 9)
        enough_en_values = [0,1]
        enough_wr_values = [0,1]

        # Create all possible combinations (324.000)
        combinations = list(itertools.product(
            wealth_values, tau_income_values, tau_wealth_values,
            [delta], [mu], [interest_rate], [investment_needed], 
            [r0_se], [r1_se], [r_mul], [q_se], [q_en], 
            v_values, enough_en_values, enough_wr_values
        ))

        state_table = pd.DataFrame(combinations, columns=state_columns)

        # Filter out rows where r1_se < r0_se
        state_table = state_table[state_table["r1_se"] >= state_table["r0_se"]]
        print(f"State_table with shape {state_table.shape} has been created.")

        # Generate rewards for each state-action pair
        training_table = state_table.copy()
        horizon_rewards = future_rewards(
            state=training_table, 
            action_dim=4, 
            horizon=3,
            beta=0.9,
            action_outcomes_fn=action_outcomes
        )
        
        training_table['action_0'] = horizon_rewards[:, 0]
        training_table['action_1'] = horizon_rewards[:, 1]
        training_table['action_2'] = horizon_rewards[:, 2]
        training_table['action_3'] = horizon_rewards[:, 3]
        training_table.to_csv('training_table', sep=";")
        print(f"Training_table with shape {training_table.shape} has been created.")

    # Normalize state columns
    scaler = StandardScaler()
    training_table[state_columns] = scaler.fit_transform(training_table[state_columns])
    joblib.dump(scaler, f"state_scaler_{mu}.pkl")
    print("Scaler saved to state_scaler.pkl")

    dataset = TrainingDataset(training_table, state_columns, reward_columns)
    print('Iterable Dataset created.')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    print('Dataloader created.')

    state_dim = len(state_columns)
    action_dim = len(reward_columns)
    policy_network = DeepQNet(state_dim, action_dim, 256)
    optimizer = optim.AdamW(policy_network.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = nn.MSELoss()
    print('Model initialized.')

    epochs = 100
    batch_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} — Training...\n")
        epoch_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):

            optimizer.zero_grad()
            predictions = policy_network(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=2.0)
            optimizer.step()

            batch_losses.append(loss.item())
            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")

        print(f"Epoch {epoch+1} completed — Avg Loss: {epoch_loss / len(dataloader):.6f}")
        scheduler.step(epoch_loss)
        torch.save(policy_network.state_dict(), f"model_paths_withGov/{model_name}")

    # Save final model
    torch.save(policy_network.state_dict(), f"model_paths_withGov/{model_name}")
    print("Training complete. Model saved.")
    
