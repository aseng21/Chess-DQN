from agent.dqn_agent import DQNAgent
from agent.models import ChessDQN
import torch

from env.env import ChessEnv

env = ChessEnv()
action_size = 4672  # Max possible moves in chess
model = ChessDQN(action_size)
agent = DQNAgent(model, action_size)

state = env.reset()
done = False

while not done:
    env.render()
    legal_moves = env.get_legal_actions()

    # Convert the state to a numpy array and then to a tensor
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    action = agent.select_action(state_tensor, legal_moves)
    state, reward, done = env.step(action)

env.render()
print("Game over!")
