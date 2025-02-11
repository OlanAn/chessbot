from backend.dql_trainer import build_dql_model, ReplayMemory, train_dql_self_play
import os.path
from keras.models import load_model
import stockfish


# Paramètres

model_to_load = None
model_to_save = 'models/ben.keras'

input_shape = (8, 8)
num_actions = 4672
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.95
batch_size = 10
num_episodes = 10
max_moves = 100

# Initialisation
if model_to_load is not None and os.path.isfile(model_to_load):
    model = load_model(model_to_load)
else:
    model = build_dql_model(input_shape=input_shape, num_actions=num_actions)

target_model = build_dql_model(input_shape=input_shape, num_actions=num_actions)
memory = ReplayMemory(capacity=10000)

#Instanciate Stockfish board
boardStockfish = stockfish.Stockfish()
print("Evaluation start : ",boardStockfish.get_evaluation().get("value"))
print()

# Entraîner le bot en jouant contre lui-même
train_dql_self_play(model, 
                    target_model, 
                    memory, 
                    num_episodes, 
                    gamma, 
                    epsilon, 
                    epsilon_decay, 
                    batch_size, 
                    max_moves, 
                    boardStockfish)

# Sauvegarder le modèle entraîné
model.save(model_to_save)
