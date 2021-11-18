from copy import deepcopy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.env_utils import Env
from src.data_gen_utils import Buffer
from src.base_model_utils import get_actor_model, get_critic_model
from src.misc_utils import logprobabilities, sample_action
from src.CONSTS import BATCH_SIZE, MIN_NUM_ATOMS, MAX_NUM_ATOMS

# hyperparameters of PPO
steps_per_epoch = 1024
epochs = 200
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-4
train_policy_iterations = 32
train_value_iterations = 32
lam = 0.97
target_kl = 0.01


# Initialize the buffer
buffer = Buffer(steps_per_epoch)

# Initialize the actor and the critic as keras models
actor = get_actor_model()
critic = get_critic_model()

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)


@tf.function
def get_logits(state):
    '''
    use tf.function decorator to speed up forward pass
    '''
    logits = actor(state)
    return logits


@tf.function
def get_value(state):
    '''
    use tf.function decorator to speed up forward pass
    '''
    value = critic(state)
    return value


@tf.function
def train_policy(observation_buffer,
                 action_buffer,
                 logprobability_buffer,
                 advantage_buffer):

    with tf.GradientTape() as tape:
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
    return value_loss


# Iterate over the number of epochs
episode_return = 0
episode_length = 0
num_atoms = np.random.randint(MIN_NUM_ATOMS, MAX_NUM_ATOMS)
env = Env(num_atoms)
state = env.reset()
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    max_episode_len = -np.inf
    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        # Get the logits, action, and take one step in the environment
        logits = actor(state[np.newaxis, ...])
        value_t = critic(state[np.newaxis, ...])
        action = sample_action(logits[0].numpy(), state)
        state_new, done, reward = env.step(action)
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(state, action, reward, value_t, logprobability_t)

        # Update the observation
        state = state_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(state[np.newaxis, ...])
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            # generate molecules with different length
            num_atoms = np.random.randint(MIN_NUM_ATOMS, MAX_NUM_ATOMS)
            env = Env(num_atoms)
            if episode_length > max_episode_len:
                max_episode_len = episode_length
            state, episode_return, episode_length = env.reset(), 0, 0

    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        arr = np.arange(steps_per_epoch)
        np.random.shuffle(arr)
        num_batches = steps_per_epoch // BATCH_SIZE
        train_batch_idx = np.array_split(arr, num_batches)
        for batch in train_batch_idx:
            kl = train_policy(
                observation_buffer[batch, ...],
                action_buffer[batch],
                logprobability_buffer[batch],
                advantage_buffer[batch]
            )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        arr = np.arange(steps_per_epoch)
        np.random.shuffle(arr)
        num_batches = steps_per_epoch // BATCH_SIZE
        train_batch_idx = np.array_split(arr, num_batches)
        for batch in train_batch_idx:
            ll = train_value_function(observation_buffer[batch, ...],
                                      return_buffer[batch])
    print("value fitting loss = {}".format(ll))

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
