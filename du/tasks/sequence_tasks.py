import numpy as np
from . import reber_grammar_tasks


def add_mask_from_lengths(datamap, max_length=None):
    """
    takes in a dict with key "lengths" and adds a key "mask"
    corresponding to those lengths
    """
    lengths = datamap["lengths"]
    if max_length is None:
        max_length = max(lengths)
    mask = np.zeros((len(lengths), max_length), dtype=lengths.dtype)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1
    datamap["mask"] = mask
    return datamap

# ################################# lag task #################################


def lag_task(lag, length, dtype):
    inputs = np.random.randint(0, 2, length).astype(dtype)
    outputs = np.array(lag * [0] + list(inputs), dtype=dtype)[:length]
    return inputs, outputs


def lag_task_minibatch(lag, length, batch_size, dtype):
    """
    returns a minibatch for the lag task - a task where one
    needs to output the input sequence shifted by a certain constant
    amount
    """
    inputs = []
    outputs = []
    for _ in range(batch_size):
        i, o = lag_task(lag, length, dtype)
        inputs.append(i)
        outputs.append(o)
    x = np.array(inputs)[..., np.newaxis]
    y = np.array(outputs)[..., np.newaxis]
    return {"x": x, "y": y}

# ################################# add task #################################


def add_task_minibatch(min_length, max_length, batch_size, dtype):
    """
    returns a minibatch for the add task - where one needs to output
    the sum of 2 elements of the first input sequence where the
    second input sequence is 1
    """
    x = np.zeros((batch_size, max_length, 2), dtype=dtype)
    x[:, :, 0] = np.random.uniform(size=(batch_size, max_length))
    lengths = np.zeros((batch_size,), dtype=dtype)
    y = np.zeros((batch_size,), dtype=dtype)
    for n in range(batch_size):
        # randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # store length
        lengths[n] = length
        # zero out x after the end of the sequence
        x[n, length:, 0] = 0
        # set the second dimension to 1 at the indices to add
        x[n, np.random.randint(length // 10), 1] = 1
        x[n, np.random.randint(length // 2, length), 1] = 1
        # multiply and sum the dimensions of x to get the target value
        y[n] = np.sum(x[n, :, 0] * x[n, :, 1])
    return {"x": x, "y": y, "lengths": lengths}

# ############################## reber grammar ##############################


def reber_grammar_minibatch(batch_size, dtype, min_length=10):
    examples = reber_grammar_tasks.get_n_examples(n=batch_size,
                                                  minLength=min_length)
    inputs, outputs = zip(*examples)
    lengths = map(len, inputs)
    max_length = max(lengths)
    x = np.zeros((batch_size, max_length, 7), dtype=dtype)
    y = np.zeros((batch_size, max_length, 7), dtype=dtype)
    for i in range(batch_size):
        x[i, :lengths[i]] = np.array(inputs[i], dtype=dtype)
        y[i, :lengths[i]] = np.array(outputs[i], dtype=dtype)
    lengths = np.array(lengths, dtype=dtype)
    return {"x": x, "y": y, "lengths": lengths}


def embedded_reber_grammar_minibatch(batch_size, dtype, min_length=10):
    examples = reber_grammar_tasks.get_n_embedded_examples(
        n=batch_size,
        minLength=min_length)
    inputs, outputs = zip(*examples)
    lengths = map(len, inputs)
    max_length = max(lengths)
    x = np.zeros((batch_size, max_length, 7), dtype=dtype)
    y = np.zeros((batch_size, max_length, 7), dtype=dtype)
    for i in range(batch_size):
        x[i, :lengths[i]] = np.array(inputs[i], dtype=dtype)
        y[i, :lengths[i]] = np.array(outputs[i], dtype=dtype)
    lengths = np.array(lengths, dtype=dtype)
    return {"x": x, "y": y, "lengths": lengths}
