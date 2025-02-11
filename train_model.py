from backend.dql_trainer import build_dql_model, ReplayMemory, train_dql_self_play
import os.path
from keras.models import load_model


# Paramètres
input_shape = (8, 8)
num_actions = 4672
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.95
batch_size = 10
num_episodes = 1

# Initialisation
model = build_dql_model(input_shape=input_shape, num_actions=num_actions)
    
#model = load_model("models/saved_model_self_play_.h5")

target_model = build_dql_model(input_shape=input_shape, num_actions=num_actions)
memory = ReplayMemory(capacity=10000)

# Entraîner le bot en jouant contre lui-même
train_dql_self_play(model, target_model, memory, num_episodes, gamma, epsilon, epsilon_decay, batch_size)

# Sauvegarder le modèle entraîné
model.save("models/saved_model_self_play_.h5")
