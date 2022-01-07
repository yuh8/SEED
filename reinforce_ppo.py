import numpy as np
import tensorflow as tf
from copy import deepcopy
from datetime import date
from src.env_utils import Env
from src.data_gen_utils import Buffer
from src.base_model_utils import (get_actor_model,
                                  get_critic_model)
from src.misc_utils import (logprobabilities, get_entropy, get_kl_divergence,
                            sample_action, save_model_to_json,
                            create_folder)
from src.CONSTS import BATCH_SIZE, MIN_NUM_ATOMS, MAX_GEN_ATOMS
today = str(date.today())

# hyperparameters of PPO
steps_per_epoch = 20480
epochs = 120
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 1e-6
value_learning_rate = 1e-5
train_policy_iterations = 2
train_value_iterations = 2
lam = 0.97
target_kl = 0.1
entropy_weight = 0.01
kl_weight = 0.01


# Initialize the buffer
buffer = Buffer(steps_per_epoch)

# Initialize the actor and the critic as keras models
actor = get_actor_model()
critic = get_critic_model()
create_folder("rl_model_{}".format(today))
save_model_to_json(actor, "rl_model_{}/rl_model.json".format(today))

# Initialize the policy and the value function optimizers
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_learning_rate)

# logs
train_log_dir = 'logs_{}/'.format(today)
writer = tf.summary.create_file_writer(train_log_dir)
tf.summary.trace_on(graph=True, profiler=False)


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
                 logits_buffer,
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
        entropy_value = get_entropy(logits)
        kl_value = get_kl_divergence(logits, logits_buffer)
        # maximizing reward and entropy while reducing KL divergence
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
            + entropy_weight * entropy_value
            - kl_weight * kl_value
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_grads, _ = tf.clip_by_global_norm(policy_grads, 0.5)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(kl_value)
    return policy_loss, kl, entropy_value


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, value_buffer, return_buffer):
    with tf.GradientTape() as tape:
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


# Training starts here
episode_return = 0
episode_length = 0
num_atoms = np.random.randint(MIN_NUM_ATOMS, MAX_GEN_ATOMS)
env = Env(num_atoms)
state = env.reset()
max_mean_return = -np.inf
policy_train_step = 0

with writer.as_default():
    for epoch in range(epochs):
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        max_episode_len = -np.inf
        for t in range(steps_per_epoch):
            # Get the logits, action, and take one step in the environment
            X_in = deepcopy(state[np.newaxis, ...])
            X_in[..., -1] /= 8
            logits = actor(X_in, training=False)
            value_t = critic(X_in, training=False)
            action = sample_action(logits[0].numpy(), state)
            state_new, done, reward = env.step(action)
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            logprobability_t = logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(state, action, reward, value_t, logprobability_t, logits)

            # Update the observation
            state = state_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                X_in = deepcopy(state[np.newaxis, ...])
                X_in[..., -1] /= 8
                last_value = 0 if done else critic(X_in, training=False)
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                # Generate molecules with different length
                num_atoms = np.random.randint(MIN_NUM_ATOMS, MAX_GEN_ATOMS)
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
            logits_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            arr = np.arange(steps_per_epoch)
            np.random.shuffle(arr)
            num_batches = steps_per_epoch // BATCH_SIZE
            train_batch_idx = np.array_split(arr, num_batches)
            for batch in train_batch_idx:
                policy_loss, kl, entropy_value = train_policy(
                    observation_buffer[batch, ...],
                    action_buffer[batch],
                    logprobability_buffer[batch],
                    logits_buffer[batch],
                    advantage_buffer[batch]
                )
                if policy_train_step == 0:
                    tf.summary.trace_export(name="neunicorn", step=policy_train_step)
                # This may fluctuate abit
                tf.summary.scalar('policy_loss', policy_loss, step=policy_train_step)
                # This should be going up to limit and then come down
                tf.summary.scalar('kl_divergence', kl, step=policy_train_step)
                # This should slowly decrease overtime
                tf.summary.scalar('entropy', tf.reduce_mean(entropy_value), step=policy_train_step)
                policy_train_step += 1

                if kl > 1.5 * target_kl:
                    # Early Stopping
                    early_stop = True
                    break
            else:
                continue
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

        # Print mean return and length for each epoch
        mean_return = np.round(sum_return / num_episodes, 3)
        tf.summary.scalar('mean_return', mean_return, step=epoch)
        writer.flush()
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
        if mean_return > max_mean_return:
            max_mean_return = mean_return
            actor.save_weights("./rl_model_{}/weights/".format(today))
