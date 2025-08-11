import jax
import jax.numpy as jnp
import pytest

from gigastep import make_scenario

# ---------- helpers ----------


@pytest.fixture
def rng():
    return jax.random.PRNGKey(3)


def _random_actions(key, env, batch_size: int | None = None):
    """
    Sample continuous actions in [-1, 1] with the same shape the env expects.
    Works for both continuous and discrete modes since the env clips/looks up.
    """
    if batch_size is None:
        shape = (env.n_agents, 3)
    else:
        shape = (batch_size, env.n_agents, 3)
    return jax.random.uniform(key, shape=shape, minval=-1.0, maxval=1.0)


# ---------- single-env tests ----------


def test_scenario_20v20_runs(rng):
    env = make_scenario("identical_20_vs_20")
    rng, key_reset = jax.random.split(rng, 2)

    obs, state = env.reset(key_reset)
    assert isinstance(obs, jnp.ndarray) or isinstance(obs, tuple)
    assert isinstance(state, tuple)

    rng, key_action, key_step = jax.random.split(rng, 3)
    action = _random_actions(key_action, env)
    obs, state, rewards, dones, ep_done = env.step(state, action, key_step)

    # basic shape checks
    assert rewards.shape == (env.n_agents,)
    assert dones.shape == (env.n_agents,)
    assert ep_done.dtype == bool


@pytest.mark.parametrize(
    "kwargs",
    [
        {"debug_reward": True},
        {"obs_type": "vector"},
        {"obs_type": "vector", "enable_waypoints": False},
    ],
)
def test_scenario_20v20_variants(rng, kwargs):
    env = make_scenario("identical_20_vs_20", **kwargs)
    rng, key_reset = jax.random.split(rng, 2)

    obs, state = env.reset(key_reset)
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = _random_actions(key_action, env)
    obs, state, rewards, dones, ep_done = env.step(state, action, key_step)

    assert rewards.shape == (env.n_agents,)
    assert dones.dtype == jnp.bool_
    assert ep_done.dtype == bool


# ---------- vmapped / batched tests ----------


def test_vmap_scenario_20v20_shapes(rng):
    batch_size = 6
    env = make_scenario("identical_20_vs_20")
    rng, key_reset = jax.random.split(rng, 2)
    key_reset = jax.random.split(key_reset, batch_size)

    obs, state = env.v_reset(key_reset)
    assert isinstance(state, tuple)
    ep_dones = jnp.zeros(batch_size, dtype=jnp.bool_)

    rng, key_action, key_step = jax.random.split(rng, 3)
    action = _random_actions(key_action, env, batch_size=batch_size)
    key_step = jax.random.split(key_step, batch_size)

    obs, state, rewards, dones, ep_dones = env.v_step(state, action, key_step)

    # shape checks (RGB obs is default; if vector/rgb_vector, user’s other tests cover that)
    assert rewards.shape == (batch_size, env.n_agents)
    assert dones.shape == (batch_size, env.n_agents)
    assert ep_dones.shape == (batch_size,)

    # Reset only done episodes (should not error and should preserve shapes)
    rng, key = jax.random.split(rng, 2)
    obs2, state2 = env.reset_done_episodes(obs, state, ep_dones, key)
    # same shapes after selective reset
    if isinstance(obs, tuple):
        assert len(obs) == len(obs2)
        for a, b in zip(obs, obs2):
            assert a.shape == b.shape
    else:
        assert obs.shape == obs2.shape
    # spot-check structure remains a (agent_state,map_state) tree
    assert isinstance(state2, tuple) and len(state2) == 2


@pytest.mark.parametrize(
    ("name", "obs_type"),
    [
        ("hide_and_seek_5_vs_5", "vector"),
        ("waypoint_5_vs_5", "vector"),
        ("identical_20_vs_20", "rgb"),
    ],
)
def test_maps_all_variants(rng, name, obs_type):
    env = make_scenario(name, obs_type=obs_type, maps="all")
    rng, key_reset = jax.random.split(rng, 2)
    obs, state = env.reset(key_reset)

    rng, key_action, key_step = jax.random.split(rng, 3)
    action = _random_actions(key_action, env)
    obs, state, rewards, dones, ep_done = env.step(state, action, key_step)

    assert rewards.shape == (env.n_agents,)
    assert dones.dtype == jnp.bool_


@pytest.mark.parametrize(
    "name",
    [
        "hide_and_seek_5_vs_5",
        "waypoint_5_vs_5",
        "waypoint_5_vs_3_fobs_rgb_maps_cont",
    ],
)
def test_other_scenarios_vmapped(rng, name):
    batch_size = 4
    env = make_scenario(name)
    rng, key_reset = jax.random.split(rng, 2)
    key_reset = jax.random.split(key_reset, batch_size)

    obs, state = env.v_reset(key_reset)
    ep_dones = jnp.zeros(batch_size, dtype=jnp.bool_)
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = _random_actions(key_action, env, batch_size=batch_size)
    key_step = jax.random.split(key_step, batch_size)

    obs, state, rewards, dones, ep_dones = env.v_step(state, action, key_step)

    assert rewards.shape == (batch_size, env.n_agents)
    assert dones.shape == (batch_size, env.n_agents)
    assert ep_dones.shape == (batch_size,)


@pytest.mark.parametrize(
    "name",
    [
        "waypoint_5_vs_5",
        "hide_and_seek_5_vs_5",
        "identical_5_vs_1_fobs_vec_void_cont",
    ],
)
def test_state_obs_scenarios_vector(rng, name):
    env = make_scenario(name, obs_type="vector")
    rng, key_reset = jax.random.split(rng, 2)

    obs, state = env.reset(key_reset)
    rng, key_action, key_step = jax.random.split(rng, 3)
    action = _random_actions(key_action, env)
    obs, state, rewards, dones, ep_done = env.step(state, action, key_step)

    # For vector obs, per-agent obs length should match spec shape[0]
    assert isinstance(obs, jnp.ndarray)
    assert obs.shape[0] == env.n_agents
    assert obs.shape[-1] == env.observation_space.shape[0]
