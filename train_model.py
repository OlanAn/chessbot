from backend.dql_trainer import build_dql_model, ReplayMemory, train_dql_self_play

# Paramètres
input_shape = (8, 8)
num_actions = 4672
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99
batch_size = 10
num_episodes = 50

# Initialisation
model = build_dql_model(input_shape=input_shape, num_actions=num_actions)
target_model = build_dql_model(input_shape=input_shape, num_actions=num_actions)
memory = ReplayMemory(capacity=10000)

# Entraîner le bot en jouant contre lui-même
train_dql_self_play(model, target_model, memory, num_episodes, gamma, epsilon, epsilon_decay, batch_size)

# Sauvegarder le modèle entraîné
model.save("models/saved_model_self_play_.h5")
