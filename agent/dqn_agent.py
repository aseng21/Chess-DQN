import random
import torch
import torch.optim as optim

class DQNAgent:
    def __init__(self, model, action_size):
        self.model = model
        self.target_model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.memory = []
        self.batch_size = 32

    def select_action(self, state, legal_moves):
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        q_values = self.model(state)
        return max(legal_moves, key=lambda move: q_values[move])

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update += self.gamma * torch.max(self.target_model(next_state))
            q_values = self.model(state)
            q_values[action] = q_update
            loss = torch.mean((q_values[action] - q_update) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
