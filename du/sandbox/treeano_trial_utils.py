import du
import canopy


def save_network(trial, name, network):
    with du.timer("serializing network (%s) for %s:%d"
                  % (name, trial.trial_name, trial.iteration_num)):
        value_dict = canopy.network_utils.to_value_dict(network)
        du.joblib_utils.dump_dir(
            value_dict, trial.file_path(name + "_values"))
        canopy.serialization.pickle_network(
            network, trial.file_path(name + "_network"))


def load_previous_network(trial_name,
                          iteration_num,
                          network=None,
                          load_previous_architecture=False,
                          trials_dir=None,
                          strict=False):
    """
    takes in a previous trial and a network, and returns a new network

    load_previous_architecture:
    whether or not to use the EXACT same architecture as the previous network
    and ignore the given network
    True => ignore given network, use same network as last time
    False => try to fit in previous network parameters into given network
             architecture

    strict:
    if the previous and given network states should match exactly
    True => make sure previous network has exact same state as given
            network, and that all state has the same shape
            use case: loading a previous network with the same architecture
    False => ignore parameters with the same name that changed shape,
             and missing parameters
             use case: wanting to load a subset of parameters from a
                       previous network

    example:
    >>> # load some subset of parameters from previous trial
    >>> network = load_previous_network("some_trial_name",
    >>>                                 1,
    >>>                                 network)
    """
    # load previous trial
    tmp_trial = du.trial.TrialState(
        trial_name,
        iteration_num,
        trials_dir=trials_dir)
    if load_previous_architecture:
        # return previous network
        return canopy.serialization.unpickle_network(
            tmp_trial.file_path("best_network"))
    else:
        assert network is not None
        # put old values into given network
        du.info("Loading previous network values: %d",
                iteration_num)
        value_dict = du.joblib_utils.load_dir(
            tmp_trial.file_path("best_values"))

        if strict:
            # use strictest settings because we want
            # to load an existing network
            canopy.network_utils.load_value_dict(
                network,
                value_dict,
                strict_keys=True,
                ignore_different_shape=False)
        else:
            # use least strict settings so we can
            # partially load some layers
            canopy.network_utils.load_value_dict(
                network,
                value_dict,
                strict_keys=False,
                ignore_different_shape=True)
