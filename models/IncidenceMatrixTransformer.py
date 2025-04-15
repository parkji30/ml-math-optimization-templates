import jax
import equinox as eqx
import jax.numpy as jnp
from typing import Callable, Sequence, Union, cast
from jaxtyping import Array

"""
An equivariant model used to train incidence (or adjacency matrix) data.
"""


def dot_product_attention_weights(
    query,
    key,
    mask,
):
    C = query.shape[0]
    logits = jnp.einsum("bij, bjk -> bik", query, key.transpose(0, 2, 1)) / jnp.sqrt(C)

    # if mask is not None:
    # # 	# We can't use 0.
    # # 	# You have to use mask because some values are going to be 0.
    # 	logits = jnp.where(mask, logits, -999999)
    # 	logits = cast(Array, logits)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits, axis=-1).astype(logits)
    return weights


def dot_product_attention(query, key, value, mask):
    weights = dot_product_attention_weights(query, key, mask)

    attn = jnp.einsum("cij, cjk->cik", weights, value)
    return attn


class CLSToken(eqx.Module):
    cls_token: jax.Array

    def __init__(self, shape, dtype=jnp.float32, key=None):
        """
        CLS Token for learning embeddings on a global scale.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # 1, E, 128 would mean you attach to node
        # N, 1, 128 would mean you attach to edge
        self.cls_token = jax.random.normal(key, shape=shape, dtype=dtype)

    def __call__(self, x, axis=1):
        """
        Append the CLS token to x.
        """
        # Node Dimension
        if axis == 1:
            cls_tokens = jnp.tile(self.cls_token, reps=(x.shape[0], 1, x.shape[2], 1))
        # We want to append on Edge
        elif axis == 2:
            cls_tokens = jnp.tile(self.cls_token, reps=(x.shape[0], x.shape[1], 1, 1))
        else:
            cls_tokens = self.cls_token
        x_with_cls = jnp.concatenate([x, cls_tokens], axis=axis)
        return x_with_cls


class IncidenceAttentionHead(eqx.Module):
    linear1: eqx.nn.Linear | Callable
    linear2: eqx.nn.Linear | Callable
    linear3: eqx.nn.Linear | Callable
    linear_out: eqx.nn.Linear | Callable

    def __init__(
        self,
        embed_dim: int,
        attention_hidden_dim: int,
        key: jax.random.PRNGKey,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.linear1 = eqx.nn.Linear(
            embed_dim, attention_hidden_dim, key=key, dtype=dtype
        )
        self.linear2 = eqx.nn.Linear(
            embed_dim, attention_hidden_dim, key=key, dtype=dtype
        )
        self.linear3 = eqx.nn.Linear(
            embed_dim, attention_hidden_dim, key=key, dtype=dtype
        )
        self.linear_out = eqx.nn.Linear(
            attention_hidden_dim, embed_dim, key=key, dtype=dtype
        )

    def __call__(
        self, x: jax.Array, mask: jax.Array, attention_along_channel: bool = False
    ) -> jax.Array:
        # Assuming x is of shape (B, N, E, C)
        _, _, C = x.shape

        # This will ensure that linear layers don't pick up on the x.
        # the bias term will not affect this.
        if mask is not None:
            # x *= mask  # This is correct
            x = jnp.where(mask, x, 0)

        query = jax.vmap(jax.vmap(self.linear1))(x)
        key = jax.vmap(jax.vmap(self.linear2))(x)
        value = jax.vmap(jax.vmap(self.linear3))(x)

        # Swaps N, E dimension
        if attention_along_channel:
            query = query.transpose(2, 0, 1)
            key = key.transpose(2, 0, 1)
            value = value.transpose(2, 0, 1)
            if mask is not None:
                mask = mask.transpose(2, 0, 1)
                # this will ensure the mask values turn 0
                # and the others stay as
                query = jnp.where(mask, query, 0)
                key = jnp.where(mask, key, 0)
                value = jnp.where(mask, value, 0)

        # This is the issue with masking
        # We divided by N before but that's batch relative
        query_key = query @ key.transpose(0, 2, 1) / jnp.sqrt(C)

        # THIS STEP IS NOT NECESSARY
        if mask is not None:
            query_key = jnp.where(0, query_key, -999999)

        # This step is fine
        # (B, E, N, N)
        attention_filter = jax.nn.softmax(query_key, axis=-1)

        # Attention Output
        attention_output = attention_filter @ value

        # THIS STEP IS NOT NECESSARY
        if mask is not None:
            attention_output = jnp.where(mask, attention_output, 0)

        # Transpose back to (B, N, E, C)
        if attention_along_channel:
            attention_output = attention_output.transpose(1, 2, 0)

        # Shrink data to original dimension
        # If this is turned off, then the projecte embeddings will continue
        attention_output = jax.vmap(jax.vmap(self.linear_out))(attention_output)

        # this should be the same dimensions as the input
        return attention_output


class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    activation: jax.nn.gelu

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        key: jax.random.PRNGKey,
        dtype: jnp.dtype = jnp.float32,
    ):
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(dim, hidden_dim, key=key1, dtype=dtype)
        self.linear2 = eqx.nn.Linear(hidden_dim, dim, key=key2, dtype=dtype)
        self.activation = jax.nn.gelu

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.vmap(jax.vmap(jax.vmap(self.linear1)))(x)
        x = self.activation(x)
        x = jax.vmap(jax.vmap(jax.vmap(self.linear2)))(x)
        return x


class TransformerAttentionBlock1D(eqx.Module):
    attention: IncidenceAttentionHead
    mlp: MLP | Callable

    def __init__(
        self,
        embed_dim: int,
        attention_hidden_dim: int,
        mlp_hidden_dim: int,
        *,
        key: jax.random.PRNGKey,
        dtype: jnp.dtype = jnp.float32,
    ):
        key1, key2 = jax.random.split(key, 2)

        self.attention = IncidenceAttentionHead(
            embed_dim, attention_hidden_dim, key=key1, dtype=dtype
        )
        self.mlp = MLP(embed_dim, mlp_hidden_dim, key=key2, dtype=dtype)

    def __call__(
        self, x: jax.Array, mask: jax.Array, attention_along_channel: bool
    ) -> jax.Array:
        # Attention block
        attn_output = self.attention(x, mask, attention_along_channel)

        # Attention Res Net
        x = x + attn_output

        # MLP block
        mlp_output = self.mlp(x)
        x = x + mlp_output

        return x


class ConstantChannelTransformerCLS(eqx.Module):
    blowup_layer: Sequence[Union[eqx.nn.Linear, Callable]]
    encoder_blocks: Sequence[TransformerAttentionBlock1D]
    ff_layer: Sequence[Union[eqx.nn.Linear, Callable]]
    node_cls_token: CLSToken
    edge_cls_token: CLSToken
    # blowdown_layer: Sequence[Union[eqx.nn.Linear, Callable]]
    node_attention_block: TransformerAttentionBlock1D
    edge_attention_block: TransformerAttentionBlock1D
    num_layers: int

    def __init__(
        self,
        attention_hidden_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey,
    ):
        self.num_layers = num_layers
        self.blowup_layer = [
            eqx.nn.Linear(3, attention_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
        ]

        self.encoder_blocks = [
            TransformerAttentionBlock1D(
                attention_hidden_dim,
                attention_hidden_dim,
                attention_hidden_dim,
                key=key,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        self.node_cls_token = CLSToken((1, 1, 1, attention_hidden_dim))
        self.edge_cls_token = CLSToken((1, 1, 1, attention_hidden_dim))

        self.node_attention_block = TransformerAttentionBlock1D(
            embed_dim=attention_hidden_dim,
            attention_hidden_dim=attention_hidden_dim,
            mlp_hidden_dim=attention_hidden_dim,
            key=key,
            dtype=dtype,
        )

        self.edge_attention_block = TransformerAttentionBlock1D(
            embed_dim=attention_hidden_dim,
            attention_hidden_dim=attention_hidden_dim,
            mlp_hidden_dim=attention_hidden_dim,
            key=key,
            dtype=dtype,
        )

        self.ff_layer = [
            eqx.nn.Linear(
                2 * attention_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype
            ),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, 1, key=key, dtype=dtype),
        ]


class IncidenceMatrixTransformer(eqx.Module):
    blowup_layer: Sequence[Union[eqx.nn.Linear, Callable]]
    encoder_blocks: Sequence[TransformerAttentionBlock1D]
    node_pool_ff_layer: Sequence[Union[eqx.nn.Linear, Callable]]
    edge_pool_ff_layer: Sequence[Union[eqx.nn.Linear, Callable]]
    threshold_ff_layer: Sequence[Union[eqx.nn.Linear, Callable]]
    num_layers: int

    def __init__(
        self,
        attention_hidden_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey,
    ):
        self.num_layers = num_layers

        self.blowup_layer = [
            eqx.nn.Linear(3, attention_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
        ]
        self.encoder_blocks = [
            TransformerAttentionBlock1D(
                attention_hidden_dim,
                attention_hidden_dim,
                attention_hidden_dim,
                key=key,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        self.node_pool_ff_layer = [
            eqx.nn.Linear(attention_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
        ]

        self.edge_pool_ff_layer = [
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
        ]

        self.threshold_ff_layer = [
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=key, dtype=dtype),
            jax.nn.gelu,
            eqx.nn.Linear(mlp_hidden_dim, 1, key=key, dtype=dtype),
        ]

    def __call__(self, x: jax.Array, mask: bool = None) -> jax.Array:
        """
        Node then Edge information.
        """
        if x.shape[0] == 1:
            mask = None
        else:
            mask = jnp.expand_dims(jnp.all(x != -9999, axis=-1), axis=-1)

        # Forward
        for block in self.blowup_layer:
            block = eqx.filter_checkpoint(block)
            if mask is not None:
                x = jnp.where(mask, x, 0)
                x = cast(Array, x)
            x = eqx.filter_vmap(eqx.filter_vmap(eqx.filter_vmap(block)))(x)

        # keep the channels constant
        for block in self.encoder_blocks:
            block = eqx.filter_checkpoint(block)
            if mask is not None:
                x = jnp.where(mask, x, 0)
                x = cast(Array, x)
            x = block(x, mask, True)

        # Forward
        for block in self.node_pool_ff_layer:
            block = eqx.filter_checkpoint(block)
            x = eqx.filter_vmap(eqx.filter_vmap(eqx.filter_vmap(block)))(x)

        x = jnp.max(x, where=mask, axis=1, initial=0)
        if mask is not None:
            mask = jnp.expand_dims(jnp.all(x != -9999, axis=-1), axis=-1)

        # Edge Pool
        for block in self.edge_pool_ff_layer:
            x = eqx.filter_vmap(eqx.filter_vmap(eqx.filter_checkpoint((block))))(x)

        # FF Threshold
        x = jnp.max(x, where=mask, axis=1, initial=0)
        for block in self.threshold_ff_layer:
            x = eqx.filter_vmap(eqx.filter_checkpoint(block))(x)

        return x


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    model = IncidenceMatrixTransformer(
        attention_hidden_dim=128, mlp_hidden_dim=1024, num_layers=5
    )

    data = jax.random.normal(key=jax.random.PRNGKey(1337), shape=(2, 25, 25, 5))

    print(model(data))
