import jax.numpy as jnp
import math
import optax
import jax
import time
from jax import jit
from tqdm import tqdm
from functools import partial


"""
A generic optimization workflow that can work on any objective functions.
"""
jax.config.update("jax_enable_x64", True)


# TODO
# I used sigmoid and logit to regularize my parameter optimization
# but you can use other regularization techniques.
@jax.jit
def custom_sigmoid(x, upper_bound=1):
    return (1 / (1 + jnp.exp(-x * 10))) * upper_bound


@jit
def custom_logit(x):
    return (1 / 10) * jnp.log(x / (1 - x))


# THIS IS THE FUNCTION YOU SHOULD OVERWRITE WITH THE FUNCTINO YOU WANT.
@jit
def objective_function_to_optimize():
    """
    This is the objective function you want to optimize.
    """
    pass

# You need to set this mapping for your parameter to index.
# This mapping should correspond to the relative position in objective_function_to_optimize
PARAM_TO_INDEX = { ... }
INDEX_TO_PARAM = { ... }


@partial(jax.jit, static_argnames=["fit_type"])
def objective_function(
    tuned_params, frozen_params, x, y, fit_type, sigmoid_dict
):
    combined_params = tuned_params | frozen_params

    combined_params = {
        k: jnp.where(sigmoid_dict.get(k) > 0, custom_sigmoid(v, sigmoid_dict.get(k)), v)
        for k, v in combined_params.items()
    }

    # This is the objective function you want to optimize.
    y_pred = objective_function_to_optimize(
        ...
    )

    # Loss landscapes
    mse_loss = optax.squared_error(y_pred, y).mean()
    cos_sim = -5 * optax.cosine_similarity(y_pred, y)

    if fit_type == 2:
        return cos_sim + mse_loss
    elif fit_type == 1:
        return cos_sim
    elif fit_type == 0:
        return mse_loss


@partial(jax.jit, static_argnames=["optimizer", "fit_type"])
def step(
    x,
    y,
    jax_tunable_dict,
    frozen_dict,
    opt_state,
    optimizer,
    fit_type,
    sigmoid_dict,
):
    value = objective_function(
        jax_tunable_dict, frozen_dict, x, y, fit_type, sigmoid_dict
    )

    grad = jax.grad(objective_function, argnums=0)(
        jax_tunable_dict, frozen_dict, x, y, fit_type, sigmoid_dict
    )

    updates, opt_state = optimizer.update(
        grad,
        opt_state,
        jax_tunable_dict,
        grad=grad,
        value=value,
        value_fn=objective_function,
        freqs=x,
        trace=y,
        frozen_params=frozen_dict,
    )

    tuneable_params = optax.apply_updates(jax_tunable_dict, updates)

    return tuneable_params, opt_state, value


class JaxBackprop:
    def __init__(
        self,
        tuneable_params,
        sigmoid_dict,
        optimizer,
        loss_fn,
        dtype="fp64",
    ):
        assert len(tuneable_params) > 1

        if dtype == "fp64":
            dtype = jnp.float64
        elif dtype == "fp32":
            dtype = jnp.float32
        else:
            raise ValueError(f"dtype {dtype} not supported")

        if loss_fn == "cos_mse":
            self.fit_type = 2
        elif loss_fn == "cos":
            self.fit_type = 1
        elif loss_fn == "mse":
            self.fit_type = 0
        else:
            raise ValueError(f"dtype {self.dtype} not supported")

        self.optimizer = optimizer
        self.dtype = dtype
        self.tuneable_params = tuneable_params
        self.sigmoid_params = sigmoid_dict

    def solve_smatrix(
        self,
        x,
        y,
        params,
        iterations=100,
        show_progress=False,
    ):
        jax_tuneable_dict = {}
        jax_sigmoid_dict = {}
        jax_frozen_dict = {}

        for p, v in params.items():
            if p in self.tuneable_params:
                if p in self.sigmoid_params:
                    v = custom_logit(v)
                    jax_sigmoid_dict[PARAM_TO_INDEX[p]] = self.sigmoid_params[p]
                else:
                    jax_sigmoid_dict[PARAM_TO_INDEX[p]] = 0
                jax_tuneable_dict[PARAM_TO_INDEX[p]] = jnp.array(
                    v, dtype=self.dtype
                )
            else:
                jax_frozen_dict[PARAM_TO_INDEX[p]] = jnp.array(
                    v, dtype=self.dtype
                )
                jax_sigmoid_dict[PARAM_TO_INDEX[p]] = 0
        opt_state = self.optimizer.init(jax_tuneable_dict)

        start = time.time()
        # Update full set of params after optimization step
        if not show_progress:
            for _ in range(iterations):
                jax_tuneable_dict, opt_state, loss = step(
                    x,
                    y,
                    jax_tuneable_dict,
                    jax_frozen_dict,
                    opt_state,
                    self.optimizer,
                    self.fit_type,
                    jax_sigmoid_dict,
                )
                if math.isnan(loss):
                    print("NaN loss, exiting")
                    break
        else:
            progress_bar = tqdm(range(iterations), desc="Processing", unit="step")
            for _ in progress_bar:
                jax_tuneable_dict, opt_state, loss = step(
                    x,
                    y,
                    jax_tuneable_dict,
                    jax_frozen_dict,
                    opt_state,
                    self.optimizer,
                    self.fit_type,
                    jax_sigmoid_dict,
                )

                if math.isnan(loss):
                    print("NaN loss, exiting")
                    break

                # Update full set of params after optimization step
                tqdm_display = {}
                for k, value in jax_tuneable_dict.items():
                    if INDEX_TO_PARAM[k] in self.sigmoid_params:
                        value = custom_sigmoid(
                            value, self.sigmoid_params[INDEX_TO_PARAM[k]]
                        )
                    tqdm_display[INDEX_TO_PARAM[k]] = f"{float(value):.8f}"
                tqdm_display["loss"] = f"{float(loss):.8f}"

                progress_bar.set_postfix(tqdm_display)

        fitted_dict = {}
        for k, value in (jax_tuneable_dict | jax_frozen_dict).items():
            if INDEX_TO_PARAM[k] in self.sigmoid_params:
                value = custom_sigmoid(value, self.sigmoid_params[INDEX_TO_PARAM[k]])
            fitted_dict[INDEX_TO_PARAM[k]] = value

        end = time.time()
        run_time = end - start

        return fitted_dict, run_time, loss