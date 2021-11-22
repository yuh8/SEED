import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.env_utils import Env
from src.data_gen_utils import Buffer
from src.base_model_utils import get_actor_model, get_critic_model
from src.misc_utils import (logprobabilities, get_entropy,
                            sample_action, save_model_to_json, create_folder)
from src.CONSTS import BATCH_SIZE, MIN_NUM_ATOMS, MAX_NUM_ATOMS

# hyperparameters of PPO
steps_per_epoch = 2**14
epochs = 500
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 1e-4
value_function_learning_rate = 1e-4
train_policy_iterations = 2
train_value_iterations = 2
lam = 0.97
target_kl = 0.01
entropy_weight = 0.01


# Initialize the buffer
buffer = Buffer(steps_per_epoch)

# Initialize the actor and the critic as keras models
actor = get_actor_model()
critic = get_critic_model()
create_folder("rl_model")
save_model_to_json(actor, "rl_model/rl_model.json")

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
        logits = actor(observation_buffer, training=True)
        ratio = tf.exp(
            logprobabilities(logits, action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
            + entropy_weight * get_entropy(logits)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_grads, _ = tf.clip_by_global_norm(policy_grads, 0.5)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, value_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        vpred = critic(observation_buffer, training=True)
        vpredclipped = value_buffer + tf.clip_by_value(vpred - value_buffer, -clip_ratio, clip_ratio)
        # Unclipped value
        vf_losses1 = tf.square(vpred - return_buffer)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - return_buffer)
        value_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_grads, _ = tf.clip_by_global_norm(value_grads, 0.5)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
    return value_loss


# Iterate over the number of epochs
episode_return = 0
episode_length = 0
num_atoms = np.random.randint(MIN_NUM_ATOMS, MAX_NUM_ATOMS)
env = Env(num_atoms)
state = env.reset()
max_mean_return = -np.inf
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    max_episode_len = -np.inf
    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        # Get the logits, action, and take one step in the environment
        logits = actor(state[np.newaxis, ...], training=False)
        value_t = critic(state[np.newaxis, ...], training=False)
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
            last_value = 0 if done else critic(state[np.newaxis, ...], training=False)
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
        value_buffer,
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
                                      value_buffer[batch],
                                      return_buffer[batch])
            # print("value fitting loss = {}".format(ll))

    # Print mean return and length for each epoch
    mean_return = np.round(sum_return / num_episodes, 3)
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
    if mean_return > max_mean_return:
        actor.save_weights("./rl_model/weights/")
