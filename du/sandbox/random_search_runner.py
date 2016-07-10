import collections
import functools
import numpy as np
import pandas as pd
import du
from du.sandbox import hyperopt_utils


def repeat_random_search(jsonl_filename,
                         hyperparameters_fn,
                         targets_fn):
    """
    hyperparameters_fn:
    function that takes no arguments and returns a dictionary from
    hyperparameter name to value

    targets_fn:
    takes in hyperparameters from hyperparameters_fn
    """
    results = []
    try:
        while True:
            hyperparameters = hyperparameters_fn()
            targets = targets_fn(**hyperparameters)
            result = dict(
                hyperparameters=hyperparameters,
                targets=targets,
            )
            du.io_utils.jsonl_append(result, jsonl_filename)
            results.append(result)
    except KeyboardInterrupt:
        pass
    return results


def write_html_report(jsonl_filename,
                      html_filename,
                      hyperparameters=None,
                      target="loss",
                      goal="minimize",
                      **kwargs):
    """
    hyperparameters:
    if None, use all of them
    """
    data = du.io_utils.jsonl_load(jsonl_filename)

    df_data = collections.defaultdict(list)

    if hyperparameters is None:
        # gather set of all hyperparameters that all rows have
        hyperparameters = functools.reduce(
            set.intersection,
            [set(row["hyperparameters"].keys()) for row in data])

    assert "loss" not in hyperparameters
    for row in data:
        try:
            # target can be a function
            loss = target(row["targets"])
        except TypeError:
            loss = row["targets"][target]
        if goal == "maximize":
            loss = -loss
        df_data["loss"].append(loss)
        for hp in hyperparameters:
            df_data[hp].append(row["hyperparameters"][hp])

    df = pd.DataFrame(df_data)

    html = hyperopt_utils.html_hyperopt_report(trials_df=df, **kwargs)
    with open(html_filename, "w") as f:
        f.write(html)

    return df


# ################################ generator ################################

class BaseGenerator(object):

    def __init__(self, key):
        self.key = key

    def generate(self):
        raise NotImplementedError

    def parse(self, generated):
        return generated


class BernoulliGenerator(BaseGenerator):

    def __init__(self, key, p):
        self.key = key
        self.p = p

    def generate(self):
        return {self.key: int(np.random.rand() < self.p)}


class UniformGenerator(BaseGenerator):

    def __init__(self, key, low, high):
        self.key = key
        self.low = low
        self.high = high

    def generate(self):
        return {self.key: float(np.random.uniform(self.low, self.high))}


class LogUniformGenerator(BaseGenerator):

    def __init__(self, key, low, high):
        self.key = key
        self.low = low
        self.high = high

    def generate(self):
        return {self.key: float(np.exp(np.random.uniform(np.log(self.low),
                                                         np.log(self.high))))}


class NormalGenerator(BaseGenerator):

    def __init__(self, key, mean, std):
        self.key = key
        self.mean = mean
        self.std = std

    def generate(self):
        return {self.key: float(np.random.normal(self.mean, self.std))}


class LogNormalGenerator(BaseGenerator):

    def __init__(self, key, mean, std):
        self.key = key
        self.mean = mean
        self.std = std

    def generate(self):
        return {self.key: float(np.exp(np.random.normal(np.log(self.mean),
                                                        np.log(self.std))))}


class IntChoiceGenerator(BaseGenerator):

    def __init__(self, key, choices):
        self.key = key
        self.choices = choices

    def generate(self):
        return {self.key: int(np.random.choice(self.choices))}


class FloatChoiceGenerator(BaseGenerator):

    def __init__(self, key, choices):
        self.key = key
        self.choices = choices

    def generate(self):
        return {self.key: float(np.random.choice(self.choices))}


class ChoiceGenerator(BaseGenerator):

    def __init__(self, key, choices):
        self.key = key
        self.choices = choices
        self.num_choices_ = len(choices)
        self.keys_ = ["%s_%d" % (key, i) for i in range(self.num_choices_)]

    def generate(self):
        idx = np.random.randint(self.num_choices_)
        return {k: i == idx for i, k in enumerate(self.keys_)}

    def parse(self, generated):
        for i, k in enumerate(self.keys_):
            val = generated.pop(k)
            if val:
                generated[self.key] = self.choices[i]
        return generated


class OrdinalChoiceGenerator(BaseGenerator):

    def __init__(self, key, choices):
        self.key = key
        self.choices = choices
        self.num_choices_ = len(choices)

    def generate(self):
        return {self.key: np.random.randint(self.num_choices_)}

    def parse(self, generated):
        idx = generated.pop(self.key)
        generated[self.key] = self.choices[idx]
        return generated


class RandomHyperparametersGenerator(object):

    """
    encodes hyperparameter generation in a way that is analyzable

    rationale:
    - need way of encoding categorical variables as one-hot encoding
    - need way to specify certain parameters are dependent on others
    """

    def __init__(self):
        self.generators = []
        self.active_fns = {}

    def set_when_active(self, key, active_fn):
        self.active_fns[key] = active_fn

    def bernoulli(self, key, p=0.5):
        self.generators.append(BernoulliGenerator(key, p))
        return self

    def uniform(self, key, low, high):
        self.generators.append(UniformGenerator(key, low, high))
        return self

    def log_uniform(self, key, low, high):
        self.generators.append(LogUniformGenerator(key, low, high))
        return self

    def normal(self, key, mean, std):
        self.generators.append(NormalGenerator(key, mean, std))
        return self

    def log_normal(self, key, mean, std):
        self.generators.append(LogNormalGenerator(key, mean, std))
        return self

    def int_choice(self, key, choices):
        self.generators.append(IntChoiceGenerator(key, choices))
        return self

    def float_choice(self, key, choices):
        self.generators.append(FloatChoiceGenerator(key, choices))
        return self

    def choice(self, key, choices):
        self.generators.append(ChoiceGenerator(key, choices))
        return self

    def ordinal_choice(self, key, choices):
        self.generators.append(OrdinalChoiceGenerator(key, choices))
        return self

    def generate(self):
        res = {}
        for g in self.generators:
            res.update(g.generate())
        return res

    def parse(self, generated):
        generated = du.AttrDict(generated)
        for g in self.generators:
            g.parse(generated)
        return generated

    def resample_one(self, generated):
        new_generated = du.AttrDict(generated)
        for g in self.generators:
            active_fn = self.active_fns.get(g.key, lambda m: True)
            if not active_fn(generated):
                new_generated.update(g.generate())
        return new_generated

    def resample(self, generateds, num_samples):
        """
        takes in list of generated hyperparameters and resamples the inactive
        hyperparameters
        """
        results = list(generateds)
        while len(results) < num_samples:
            idx = np.random.randint(len(generateds))
            generated = generateds[idx]
            resampled = self.resample_one(generated)
            results.append(resampled)
        return results
