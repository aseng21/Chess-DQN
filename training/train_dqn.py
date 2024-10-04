from agent.dqn_agent import DQNAgent
from agent.models import ChessDQN
import torch

from env.env import ChessEnv

env = ChessEnv()
action_size = 4672  # Upper bound on chess move space
model = ChessDQN(action_size)
agent = DQNAgent(model, action_size)

episodes = 10000

for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        legal_moves = env.get_legal_actions()
        action = agent.select_action(torch.tensor([state], dtype=torch.float32), legal_moves)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
    agent.update_target_network()
    agent.decay_epsilon()
    print(f"Episode {episode+1} completed.")
