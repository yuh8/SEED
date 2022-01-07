import numpy as np
import tensorflow as tf
from datetime import date
from src.env_utils import Env
from src.data_gen_utils import Buffer
from src.base_model_utils import (get_actor_model,
                                  get_critic_model)
from src.misc_utils import (logprobabilities, get_entropy, get_kl_divergence,
                            sample_action, save_model_to_json,
                            create_folder)
from src.CONSTS import BATCH_SIZE, MIN_NUM_ATOMS, MAX_GEN_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH
today = str(date.today())


strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="/device:CPU:0"))
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
NUM_GPUS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_GPUS

# hyperparameters of PPO
steps_per_epoch = 40960
epochs = 120
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 1e-4
value_learning_rate = 1e-4
train_policy_iterations = 2
train_value_iterations = 2
lam = 0.97
target_kl = 0.01
entropy_weight = 0.01
kl_weight = 0.01


# Initialize the buffer
buffer = Buffer(steps_per_epoch)

# Initialize the actor and the critic as keras models
with strategy.scope():
    actor = get_actor_model()
    critic = get_critic_model()
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_learning_rate)

create_folder("rl_model_{}".format(today))
save_model_to_json(actor, "rl_model_{}/rl_model.json".format(today))

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


def train_policy(input):
    (observation_buffer,
     action_buffer,
     logprobability_buffer,
     logits_buffer,
     advantage_buffer) = input

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
        per_sample_loss = -(
            tf.minimum(ratio * advantage_buffer, min_advantage)
            + entropy_weight * entropy_value
            - kl_weight * kl_value
        )
        policy_loss = tf.nn.compute_average_loss(
            per_sample_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_grads, _ = tf.clip_by_global_norm(policy_grads, 0.5)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))
    return policy_loss, kl_value, entropy_value


# Train the value function by regression on mean-squared error
def train_value_function(input):
    observation_buffer, value_buffer, return_buffer = input
    with tf.GradientTape() as tape:
        vpred = critic(observation_buffer, training=True)
        vpredclipped = value_buffer + tf.clip_by_value(vpred - value_buffer, -clip_ratio, clip_ratio)
        # Unclipped value
        vf_losses1 = tf.square(vpred - return_buffer)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - return_buffer)
        per_batch_loss = tf.maximum(vf_losses1, vf_losses2)
        value_loss = tf.nn.compute_average_loss(
            per_batch_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_grads, _ = tf.clip_by_global_norm(value_grads, 0.5)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
    return value_loss


@tf.function
def distributed_train_policy(iter):
    results = strategy.run(train_policy, args=(iter,))
    return (strategy.reduce(tf.distribute.ReduceOp.SUM, results[0],
                            axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, results[1],
                            axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, results[2],
                            axis=None))


@tf.function
def distributed_train_value(iter):
    per_replica_losses = strategy.run(train_value_function, args=(iter,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)


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
            logits = actor(state[np.newaxis, ...], training=False)
            value_t = critic(state[np.newaxis, ...], training=False)
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
                last_value = 0 if done else critic(state[np.newaxis, ...], training=False)
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

        def policy_train_data_gen():
            while True:
                arr = np.arange(steps_per_epoch)
                np.random.shuffle(arr)
                num_batches = steps_per_epoch // GLOBAL_BATCH_SIZE
                train_batch_idx = np.array_split(arr, num_batches)
                for batch in train_batch_idx:
                    yield (observation_buffer[batch, ...],
                           action_buffer[batch, ...],
                           logprobability_buffer[batch, ...],
                           logits_buffer[batch, ...],
                           advantage_buffer[batch, ...],)

        def value_train_data_gen():
            while True:
                arr = np.arange(steps_per_epoch)
                np.random.shuffle(arr)
                num_batches = steps_per_epoch // GLOBAL_BATCH_SIZE
                train_batch_idx = np.array_split(arr, num_batches)
                for batch in train_batch_idx:
                    yield (observation_buffer[batch, ...],
                           value_buffer[batch, ...],
                           return_buffer[batch, ...])

        policy_dataset = tf.data.Dataset.from_generator(policy_train_data_gen,
                                                        output_types=(tf.float32, tf.int32, tf.float32, tf.float32, tf.float32),
                                                        output_shapes=(
                                                            tf.TensorShape([None, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1]),
                                                            tf.TensorShape(None),
                                                            tf.TensorShape(None),
                                                            tf.TensorShape(None),
                                                            tf.TensorShape(None)))
        value_dataset = tf.data.Dataset.from_generator(value_train_data_gen,
                                                       output_types=(tf.float32, tf.float32, tf.float32),
                                                       output_shapes=(
                                                           tf.TensorShape([None, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1]),
                                                           tf.TensorShape(None),
                                                           tf.TensorShape(None)))
        dist_policy_dataset = strategy.experimental_distribute_dataset(policy_dataset)
        policy_iterator = iter(dist_policy_dataset)

        dist_value_dataset = strategy.experimental_distribute_dataset(value_dataset)
        value_iterator = iter(dist_value_dataset)

        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            arr = np.arange(steps_per_epoch)
            num_batches = steps_per_epoch // GLOBAL_BATCH_SIZE
            train_batch_idx = np.array_split(arr, num_batches)
            for batch in train_batch_idx:
                policy_loss, kl, entropy_value = distributed_train_policy(
                    next(policy_iterator),
                )
                if policy_train_step == 0:
                    tf.summary.trace_export(name="neunicorn", step=policy_train_step)
                # This may fluctuate abit
                tf.summary.scalar('policy_loss', policy_loss, step=policy_train_step)
                # This should be going up to limit and then come down
                tf.summary.scalar('kl_divergence', tf.reduce_mean(kl) / NUM_GPUS, step=policy_train_step)
                # This should slowly decrease overtime
                tf.summary.scalar('entropy', tf.reduce_mean(entropy_value) / NUM_GPUS, step=policy_train_step)
                policy_train_step += 1

            #     if kl > 1.5 * target_kl:
            #         # Early Stopping
            #         early_stop = True
            #         break
            # else:
            #     continue
            # break

        # Update the value function
        for _ in range(train_value_iterations):
            arr = np.arange(steps_per_epoch)
            num_batches = steps_per_epoch // BATCH_SIZE
            train_batch_idx = np.array_split(arr, num_batches)
            for batch in train_batch_idx:
                ll = distributed_train_value(next(value_iterator),)

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
