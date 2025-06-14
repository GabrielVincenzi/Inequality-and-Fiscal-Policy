import torch
import torch.nn as nn
import torch.nn.functional as F

def temp_multiplier(predicted_actions):
    mean_val = torch.mean(predicted_actions).item()
    num_digits = len(str(int(mean_val)))
    return 2 * 10 ** (num_digits - 1)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted_rewards, all_rewards):
        # max_reward = torch.max(all_rewards)
        # target = (all_rewards == max_reward).float()  # 1 for max reward, 0 otherwise
        target = torch.nn.functional.softmax(all_rewards, dim=-1)
        loss = torch.mean((predicted_rewards - target) ** 2)  # MSE with custom target
        print(f'Target: {target}')
        return loss
    

class SmoothedParamLoss(nn.Module):
    def __init__(self, entropy_weight=0.05, temperature=1, noise_std=0.05, label_smoothing=0.05):
        """
        entropy_weight: scales the entropy bonus.
        temperature: used for softening the reward-derived target.
        noise_std: standard deviation for noise injection in logit space.
        label_smoothing: smoothing factor applied to the target distribution.
        """
        super(SmoothedParamLoss, self).__init__()
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        self.noise_std = noise_std
        self.label_smoothing = label_smoothing

    def forward(self, predicted_rewards, target_rewards, step_count):
        action_target = F.softmax(target_rewards / self.temperature, dim=-1)
        num_actions = action_target.size(-1)
        action_target = (1 - self.label_smoothing) * action_target + self.label_smoothing / num_actions

        noisy_actions = predicted_rewards * (1 + torch.randn_like(predicted_rewards) * self.noise_std)
        denom = max(self.temperature* temp_multiplier(predicted_rewards), 1e-2) #
        noisy_action_pred = F.softmax(noisy_actions / denom, dim=-1)
        
        # Compute KL divergence (using noisy predictions vs. the action_target)
        kl_loss = F.kl_div(torch.log(noisy_action_pred + 1e-8), action_target, reduction='batchmean')
        
        # Compute the entropy of the noisy predictions
        safe_probs = noisy_action_pred.clamp(min=1e-6)
        entropy = -torch.sum(safe_probs * torch.log(safe_probs), dim=-1).mean()
        loss = kl_loss - self.entropy_weight * entropy

        #if step_count % 5 == 0:
        #    print(f'Target actions: {action_target}')
        #    print(f'Noisy pred actions: {noisy_action_pred}')
        return loss
    

# Inherenty peaked target distribution
    