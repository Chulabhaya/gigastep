import hydra
import jax
import jax.numpy as jnp
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from jax import tree
from matplotlib import animation
from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    Params,
)
from mava.utils.checkpointing import Checkpointer
from mava.utils.make_env import _build_enemy_actor_from_config
from mava.utils.network_utils import get_action_head
from mava.wrappers import (
    AgentIDWrapper,
    AutoResetWrapper,
    LearnedEnemyGigastepWrapper,
)
from omegaconf import DictConfig, OmegaConf

from gigastep import ScenarioBuilder


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype == np.uint8:
        return frames
    if np.issubdtype(frames.dtype, np.floating):
        frames = np.clip(frames, 0.0, 1.0)
        return (frames * 255.0 + 0.5).astype(np.uint8)
    return np.clip(frames, 0, 255).astype(np.uint8)


def save_animation(
    frames: np.ndarray,
    path: str = "episode.gif",  # ".mp4" also works
    fps: int = 30,
    start_t: int = 0,  # used only if dones=None
    dpi: int = 100,
    dones: np.ndarray | None = None,  # <-- add this
):
    frames = _to_uint8(frames)
    T, H, W, C = frames.shape
    assert C == 3

    # Build per-frame labels
    if dones is None:
        labels = np.arange(start_t, start_t + T, dtype=int)
    else:
        dones = np.asarray(dones, dtype=bool)
        assert dones.shape == (T,), f"`dones` must be shape ({T},), got {dones.shape}"
        labels = np.empty(T, dtype=int)
        counter = 0
        for i in range(T):
            # show 0 on frames where done[i] is True; otherwise keep counting
            labels[i] = 0 if dones[i] else counter
            counter = 0 if dones[i] else (counter + 1)

    # Figure exactly sized to the frame, no padding/margins
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire figure
    ax.set_axis_off()
    im = ax.imshow(frames[0], animated=True, interpolation="nearest", aspect="equal")

    # readable timestep label with black outline
    txt = ax.text(
        0.02,
        0.98,
        f"t={labels[0]:04d}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )

    def init():
        im.set_data(frames[0])
        txt.set_text(f"t={labels[0]:04d}")
        return im, txt

    def update(i):
        im.set_data(frames[i])
        txt.set_text(f"t={labels[i]:04d}")
        return im, txt

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=T, interval=1000 / fps, blit=True
    )

    # choose writer based on extension; also ensure no extra padding on save
    ext = path.split(".")[-1].lower()
    if ext in {"mp4", "m4v", "mov"}:
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=2000, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
        )
        anim.save(path, writer=writer, dpi=dpi, savefig_kwargs={"pad_inches": 0})
    elif ext == "gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(path, writer=writer, dpi=dpi, savefig_kwargs={"pad_inches": 0})
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported extension: .{ext}")

    plt.close(fig)
    return path


def render(config: DictConfig):
    # --- Initialize environment --- #
    scenario = ScenarioBuilder.from_config(config.env.scenario.task_config)
    base_env = scenario.make(**config.env.kwargs)

    enemy_actor_apply_fn, enemy_actor_params, hdim = _build_enemy_actor_from_config(
        config, base_env, add_global_state=True
    )

    env = AutoResetWrapper(
        AgentIDWrapper(
            LearnedEnemyGigastepWrapper(
                base_env,
                enemy_actor_apply_fn,
                enemy_actor_params,
                hdim,
                ScannedRNN.initialize_carry,
                False,
                True,
            )
        )
    )

    # --- Initialize actor and critic --- #
    # We don't actually need the critic for inference, just doing it right now
    # because of the existing checkpoint loading process expecting both networks
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)
    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=config.network.hidden_state_dim,
    )
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        centralised_critic=True,
    )

    # Initialise observation with obs of all agents.
    num_agents = env.num_agents
    init_obs = env.observation_spec.generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(
            x[jnp.newaxis, ...], 1, axis=0
        ),  # Extra 1 is for batch dimension over # of envs
        init_obs,
    )
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, 1, num_agents), dtype=bool)
    init_obs_done = (init_obs, init_done)

    # Initialise hidden state.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (1, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (1, num_agents), config.network.hidden_state_dim
    )
    key, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=3
    )
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_obs_done)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_obs_done)
    params = Params(actor_params, critic_params)

    # --- Load pre-trained weights of actor --- #
    config.logger.system_name = "rec_mappo"
    # Need to provide specific checkpoint UID to load from
    loaded_checkpoint = Checkpointer(
        model_name=config.logger.system_name,
        rel_dir="checkpoints",
        checkpoint_uid="fixed_probability_cone_180_view_depth_2.45_damage_depth_1.0_selfplay",
    )
    # Restore the learner state from the checkpoint
    restored_params, _ = loaded_checkpoint.restore_params(
        input_params=params, restore_hstates=False, THiddenState=HiddenStates
    )
    actor_params = restored_params.actor_params

    # --- Run an episode with loaded policy --- #
    key, env_key, learner_key = jax.random.split(key, 3)
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        env_key[jnp.newaxis, :],
    )
    init_frame = jax.vmap(env.get_global_observation, in_axes=(0))(env_states.state)
    init_done = jnp.zeros((1, num_agents), dtype=bool)
    learner_state = (
        actor_params,
        learner_key,
        env_states,
        timesteps,
        init_done,
        init_policy_hstate,
        init_frame,
    )

    def _env_step(learner_state, _):
        """Step the environment."""
        (actor_params, key, env_state, last_timestep, last_done, last_hstates, last_frame) = (
            learner_state
        )

        key, policy_key = jax.random.split(key)

        # Add a batch and time dimension to the observation.
        batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
        ac_in = (batched_observation, last_done[jnp.newaxis, :])

        # Run the network.
        policy_hidden_state, actor_policy = actor_network.apply(actor_params, last_hstates, ac_in)

        # Sample action from the policy and squeeze out the batch dimension.
        action = actor_policy.sample(seed=policy_key)

        action = action.squeeze(0)

        # Step the environment.
        env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)
        frame = jax.vmap(env.get_global_observation, in_axes=(0))(env_state.state)

        done = timestep.last().repeat(env.num_agents).reshape(1, -1)
        hstates = policy_hidden_state
        learner_state = (actor_params, key, env_state, timestep, done, hstates, frame)
        metrics = (last_frame, last_done)
        return learner_state, metrics

    def _run_episode(learner_state):
        learner_state, metrics = jax.lax.scan(_env_step, learner_state, None, 500)
        return metrics

    run_episode_jit = jax.jit(_run_episode)
    frames, dones = run_episode_jit(learner_state)
    frames = np.asarray(frames.squeeze(1))  # Get rid of fake batch dimension
    dones = np.asarray(dones.squeeze(1))[
        :, 0
    ]  # Get done of just single agent since there's a shared environment done
    _ = save_animation(frames, "episode.gif", fps=30, dpi=100, dones=dones)


@hydra.main(
    config_path="/Mava/mava/configs/default",
    config_name="rec_mappo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run renderer
    render(cfg)


if __name__ == "__main__":
    hydra_entry_point()
