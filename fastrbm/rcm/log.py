def build_log_string_train(
    train_ll: float,
    test_ll: float,
    n_iter: int,
    mean_q: float,
    grad_vbias_norm: float,
    curr_lr: float,
    count: int,
) -> str:
    """Builds a formatted string for log output during the training of the RCM

    Parameters
    ----------
    train_ll : float
        Log-likelihood on the training set.
    test_ll : float
        Log-likelihood on the testing set.
    n_iter : int
        Current number of iterations-
    mean_q : float
        Mean value of the hyperplanes weights.
    grad_vbias_norm : float
        Norm of the gradient for the visible bias.
    curr_lr : float
        Current learning rate.
    count : int
        Number of times the learning rate was augmented since last log.

    Returns
    -------
    str
        Formatted string.
    """
    train_ll_str = f"{train_ll:.3f}"
    test_ll_str = f"{test_ll:.3f}"
    mean_q_str = f"{mean_q:.5f}"
    grad_vbias_norm_str = f"{grad_vbias_norm:.6f}"
    curr_lr_str = f"{curr_lr:.6f}"
    count_lr_str = f"{count:<6}"
    return f" {n_iter:>8} | {train_ll_str:^8} | {test_ll_str:^8} | {mean_q_str:^8} | {grad_vbias_norm_str:^8} | {curr_lr_str:^8} | {count_lr_str:^8}"
