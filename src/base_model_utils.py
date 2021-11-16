import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .embed_utils import conv2d_block, res_block
from .misc_utils import load_json_model
from .CONSTS import (ATOM_LIST, CHARGES, BOND_NAMES,
                     MAX_NUM_ATOMS, FEATURE_DEPTH,
                     NUM_FILTERS, FILTER_SIZE, NUM_RES_BLOCKS)


def core_model():
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1]
    X = layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    num_loc_bond_actions = (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)
    # X_mask = layers.Input(shape=(num_act_charge_actions + num_loc_bond_actions))

    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, NUM_FILTERS]
    out = conv2d_block(X, NUM_FILTERS, FILTER_SIZE)

    # [BATCH, MAX_NUM_ATOMS/16, MAX_NUM_ATOMS/16, NUM_FILTERS]
    major_block_size = NUM_RES_BLOCKS // 4
    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    out = layers.GlobalMaxPooling2D()(out)
    action_logits = layers.Dense(num_act_charge_actions + num_loc_bond_actions,
                                 activation=None,
                                 use_bias=False)(out)
    return X, action_logits


def get_metrics():
    train_act_acc = tf.keras.metrics.CategoricalAccuracy(name="train_act_acc")
    val_act_acc = tf.keras.metrics.CategoricalAccuracy(name="val_act_acc")
    train_loss = tf.keras.metrics.CategoricalCrossentropy(name='train_loss',
                                                          from_logits=True)
    val_loss = tf.keras.metrics.CategoricalCrossentropy(name='val_loss',
                                                        from_logits=True)
    return train_act_acc, val_act_acc, train_loss, val_loss


def loss_func(y_true, y_pred):
    loss_obj = keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = loss_obj(y_true, y_pred)
    return loss


def get_optimizer(finetune=False):
    lr = 0.001
    if finetune:
        lr = 0.00001
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [200000, 400000, 600000], [lr, lr / 10, lr / 50, lr / 100],
        name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    return opt_op


class SeedGenerator(keras.Model):
    def compile(self, optimizer, loss_fn, metric_fn):
        super(SeedGenerator, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_act_acc, self.val_act_acc, \
            self.train_loss, self.val_loss = metric_fn()

    def train_step(self, train_data):
        X, y = train_data

        # capture the scope of gradient
        with tf.GradientTape() as tape:
            logits = self(X, training=True)
            loss = self.loss_fn(y, logits)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # compute metrics keeping an moving average
        self.train_act_acc.update_state(y, logits)
        self.train_loss.update_state(y, logits)
        return {"train_act_acc": self.train_act_acc.result(),
                "train_loss": self.train_loss.result()}

    def test_step(self, val_data):
        X, y = val_data

        # predict
        logits = self(X, training=False)
        # compute metrics stateless
        self.val_act_acc.update_state(y, logits)
        self.val_loss.update_state(y, logits)
        return {"val_act_acc": self.val_act_acc.result(),
                "val_loss": self.val_loss.result()}

    @property
    def metrics(self):
        # clear metrics after every epoch
        return [self.train_act_acc, self.val_act_acc,
                self.train_loss, self.val_loss]


def load_base_model():
    base_model = load_json_model("base_model/generator_model.json", SeedGenerator, "SeedGenerator")
    base_model.compile(optimizer=get_optimizer(),
                       loss_fn=loss_func,
                       metric_fn=get_metrics)
    base_model.load_weights("./base_model/weights/")
    return base_model


def get_critic_model():
    base_model = load_base_model()
    out = base_model.layers[-1].output
    # add another prediction layer
    out = layers.Dense(32, name='critic_feature')(out)
    value = layers.Dense(1, activation=None, name='critic_value')(out)
    critic = keras.Model(inputs=base_model.input, outputs=value)
    return critic


def get_actor_model():
    base_model = load_base_model()
    out = base_model.layers[-1].output
    actor = keras.Model(inputs=base_model.input, outputs=out)
    return actor
