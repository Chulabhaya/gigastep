from functools import partial

import jax
import jax.numpy as jnp
import type_enforced

from gigastep.builtin_maps import get_builtin_maps, prerender_maps
from gigastep.jax_utils import Box, Discrete


def stack_agents(*args):
    """Stack agents along the first axis."""
    num_agents = len(args)
    agents = []
    for agent in args:
        agent = dict(agent)
        agent["tracked"] = jnp.zeros((num_agents,), dtype=jnp.float32)
        agents.append(agent)
    agents = jax.tree_map(lambda *xs: jnp.stack(xs, axis=0), *agents)

    map_state = {
        "map_idx": jnp.int32(0),
        "waypoint_location": jnp.zeros(4),
        "waypoint_enabled": jnp.float32(0),
        "aux_rewards_factor": jnp.float32(1),
        "episode_length": jnp.int32(0),
    }
    return (agents, map_state)


def draw_all_agents(obs, x, y, z, teams, alive, sprite):
    """Used for the global observation"""
    team1 = (teams == 1) * 255 * alive
    team2 = (teams == 0) * 255 * alive
    r = team1
    b = team2
    g = (z * alive).astype(jnp.uint8)  # // 2
    color = jnp.stack([r, g, b], axis=-1)
    sprite1_color = color * ((sprite & 1)[:, None] > 0)
    sprite2_color = color * ((sprite & 2)[:, None] > 0)
    sprite3_color = color * ((sprite & 4)[:, None] > 0)
    # Draw agents with different sprite
    obs = obs.at[x, y].max(sprite1_color, mode="drop")
    obs = obs.at[x + 1, y].max(sprite2_color, mode="drop")
    obs = obs.at[x - 1, y].max(sprite2_color, mode="drop")
    obs = obs.at[x, y + 1].max(sprite2_color, mode="drop")
    obs = obs.at[x, y - 1].max(sprite2_color, mode="drop")
    obs = obs.at[x + 1, y + 1].max(sprite3_color, mode="drop")
    obs = obs.at[x + 1, y - 1].max(sprite3_color, mode="drop")
    obs = obs.at[x - 1, y + 1].max(sprite3_color, mode="drop")
    obs = obs.at[x - 1, y - 1].max(sprite3_color, mode="drop")

    return obs


def draw_agents_from_ego(obs, x, y, z, teams, seen, agent_id, sprite):
    """Draws the agents from the perspective of the agent with id agent_id."""
    team1 = (teams != teams[agent_id]) * 255 * seen
    team2 = (teams == teams[agent_id]) * 255 * seen
    seen_or_same_team = (seen > 0) | (teams == teams[agent_id])

    is_ego = jnp.arange(x.shape[0]) == agent_id
    r = team1 + is_ego * (255 - z)
    b = team2 * (1 - is_ego) + is_ego * (255 - z)
    g = z * seen * seen_or_same_team * (1 - is_ego) + 255 * is_ego
    color = jnp.stack([r, g, b], axis=-1)
    sprite1_color = color * ((sprite & 1)[:, None] > 0)
    sprite2_color = color * ((sprite & 2)[:, None] > 0)
    sprite3_color = color * ((sprite & 4)[:, None] > 0)

    # reset color at center of ego
    obs = obs.at[x[agent_id], y[agent_id], :].set(jnp.zeros(3, dtype=jnp.uint8))

    # Draw agents with different sprite
    obs = obs.at[x, y].max(sprite1_color, mode="drop")
    obs = obs.at[x + 1, y].max(sprite2_color, mode="drop")
    obs = obs.at[x - 1, y].max(sprite2_color, mode="drop")
    obs = obs.at[x, y + 1].max(sprite2_color, mode="drop")
    obs = obs.at[x, y - 1].max(sprite2_color, mode="drop")
    obs = obs.at[x + 1, y + 1].max(sprite3_color, mode="drop")
    obs = obs.at[x + 1, y - 1].max(sprite3_color, mode="drop")
    obs = obs.at[x - 1, y + 1].max(sprite3_color, mode="drop")
    obs = obs.at[x - 1, y - 1].max(sprite3_color, mode="drop")

    return obs


class GigastepEnv:
    def __init__(
        self,
        very_close_cone_depth=1.0,
        cone_depth=2.45,
        cone_angle=jnp.pi,
        damage_cone_depth=1.0,
        damage_cone_angle=jnp.pi,
        damage_per_second=1,
        min_tracking_time=3,
        max_tracking_time=10,
        communication_range=8,
        healing_per_second=0.1,
        use_stochastic_obs=True,
        use_stochastic_comm=True,
        enable_waypoints=False,
        collision_range=0.4,
        collision_altitude=0.5,
        limit_x=10,
        limit_y=10,
        waypoint_size=0.6,
        resolution_x=84,
        resolution_y=84,
        n_agents=10,
        maps="empty",
        per_agent_sprites=None,
        per_agent_thrust=None,
        per_agent_max_health=None,
        per_agent_range=None,
        per_agent_damage_range=None,
        per_agent_team=None,
        per_agent_idle_reward=None,
        reward_game_won=10,
        reward_defeat_one_opponent=100,
        reward_detection=0,
        reward_damage=0,
        reward_idle=0,
        reward_agent_disabled=0,
        reward_collision_agent=0,
        reward_collision_obstacle=0,
        reward_hit_waypoint=0,
        divide_reward_by_team_size=True,
        discrete_actions=False,
        obs_type="rgb",
        max_agent_in_vec_obs=15,
        max_episode_length=500,
        episode_ends_one_team_dead=True,
        jit=True,
        debug_reward=False,
        precision=jnp.float32,
        enable_view_cone_overlay=False,
        cone_detect_prob=0.60,
    ):
        self.n_agents = n_agents
        self.very_close_cone_depth = jnp.square(very_close_cone_depth)
        self.cone_depth = jnp.square(cone_depth)
        self.damage_cone_depth = jnp.square(damage_cone_depth)
        self.cone_angle = cone_angle
        self.min_tracking_time = min_tracking_time
        self.max_tracking_time = max_tracking_time
        self.damage_cone_angle = damage_cone_angle
        self.damage_per_second = damage_per_second
        self.healing_per_second = healing_per_second
        self.collision_range = jnp.square(collision_range)
        self.collision_altitude = collision_altitude
        self.debug_reward = debug_reward
        self.use_stochastic_obs = use_stochastic_obs
        self.use_stochastic_comm = use_stochastic_comm
        self.max_communication_range = communication_range
        self.waypoint_size = waypoint_size
        self.enable_waypoints = enable_waypoints
        self.max_agent_in_vec_obs = min(self.n_agents, max_agent_in_vec_obs)
        self.max_episode_length = max_episode_length
        self.episode_ends_one_team_dead = episode_ends_one_team_dead
        self.reward_game_won = reward_game_won
        self.reward_defeat_one_opponent = reward_defeat_one_opponent
        self.reward_detection = reward_detection
        self.reward_damage = reward_damage
        self.reward_idle = reward_idle
        self.reward_agent_disabled = reward_agent_disabled
        self.reward_collision_agent = reward_collision_agent
        self.reward_collision_obstacle = reward_collision_obstacle
        self.reward_hit_waypoint = reward_hit_waypoint
        self.divide_reward_by_team_size = divide_reward_by_team_size
        self.precision = precision
        self.cone_detect_prob = jnp.asarray(cone_detect_prob, dtype=self.precision)

        # --- Used for visualization ---
        self.enable_view_cone_overlay = enable_view_cone_overlay

        if per_agent_sprites is None:
            per_agent_sprites = jnp.ones(n_agents, dtype=jnp.int32)
        self._per_agent_sprites = per_agent_sprites
        if per_agent_thrust is None:
            per_agent_thrust = jnp.ones(n_agents, dtype=jnp.float32)
        self._per_agent_thrust = per_agent_thrust
        if per_agent_max_health is None:
            per_agent_max_health = jnp.ones(n_agents, dtype=jnp.float32)
        self._per_agent_max_health = per_agent_max_health
        if per_agent_range is None:
            per_agent_range = jnp.ones(n_agents, dtype=jnp.float32)
        self._per_agent_range = per_agent_range
        if per_agent_damage_range is None:
            per_agent_damage_range = jnp.ones(n_agents, dtype=jnp.float32)
        self._per_agent_damage_range = per_agent_damage_range
        if per_agent_idle_reward is None:
            per_agent_idle_reward = jnp.zeros(n_agents, dtype=jnp.float32)
        self._per_agent_idle_reward = per_agent_idle_reward

        if obs_type not in ("rgb", "vector", "rgb_vector"):
            raise ValueError(f"Unknown obs_type {obs_type} (options: rgb, vector, rgb_vector)")
        self._obs_type = obs_type
        if per_agent_team is None:
            team_blue = self.n_agents // 2
            team_red = self.n_agents - team_blue
            per_agent_team = jnp.concatenate(
                [jnp.ones((team_blue,)), jnp.zeros((team_red,))], axis=0
            )
        self._per_agent_team = per_agent_team
        n_ego_agents = 0
        n_opponents = 0
        for i in self._per_agent_team:
            if i == 0:
                n_ego_agents += 1
            else:
                n_opponents += 1
        self.n_teams = [n_ego_agents, n_opponents]
        self.n_agents_team1 = n_ego_agents
        self.n_agents_team2 = n_opponents

        self.limits = (limit_x, limit_y)
        self.z_max = 10
        self.resolution = (resolution_x, resolution_y)
        self.time_delta = 0.1

        self._maps_names = maps
        self._maps = get_builtin_maps(maps, self.limits)
        # maps need to be prerendered because range indexing is not supported in jax.jit
        self._prerendered_maps = prerender_maps(self._maps["boxes"], self.resolution, self.limits)

        # How many variables we need to represent the map
        rgb_obs_spec = Box(
            low=jnp.zeros([self.resolution[0], self.resolution[1], 3], dtype=jnp.uint8),
            high=255 * jnp.ones([self.resolution[0], self.resolution[1], 3], dtype=jnp.uint8),
        )
        # 9 variables per agents
        obs_var_count = 9 * self.max_agent_in_vec_obs
        if self.enable_waypoints:
            #  5 for the waypoint (location + active flag
            obs_var_count += 5
        if self._maps_names != "empty":
            # 4 variables per obstacle (location)
            obs_var_count += self._maps["boxes"][0].shape[0] * 4

        vector_obs_spec = Box(
            low=-jnp.inf
            * jnp.ones(
                obs_var_count,
                dtype=jnp.float32,
            ),
            high=jnp.inf
            * jnp.ones(
                obs_var_count,
                dtype=jnp.float32,
            ),
        )
        if obs_type == "rgb":
            self.observation_space = rgb_obs_spec
        elif obs_type == "vector":
            self.observation_space = vector_obs_spec
        elif obs_type == "rgb_vector":
            self.observation_space = (rgb_obs_spec, vector_obs_spec)

        self.discrete_actions = discrete_actions
        self.action_lut = jnp.array(
            jnp.meshgrid(
                jnp.array([-1.0, 0.0, 1.0]),
                jnp.array([-1.0, 0.0, 1.0]),
                jnp.array([-1.0, 0.0, 1.0]),
            )
        ).T.reshape(-1, 3)
        if self.discrete_actions:
            # 3x3x3 = 27 actions {+1, 0, -1}^3
            self.action_space = Discrete(3**3)
        else:
            self.action_space = Box(low=-jnp.ones(3), high=jnp.ones(3))

        self._v_step_agents = jax.vmap(self._step_agents)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))
        self.v_get_global_observation = jax.jit(jax.vmap(self.get_global_observation))
        self.v_set_aux_reward_factor = jax.vmap(self.set_aux_reward_factor, in_axes=(0, None))
        if jit:
            self.v_reset = jax.jit(self.v_reset)
            self.v_step = jax.jit(self.v_step)

    @property
    def teams(self):
        return self._per_agent_team

    @property
    def n_actions(self):
        if not self.discrete_actions:
            raise ValueError("n_actions is only defined for discrete actions")
        return self.action_space.n

    @partial(jax.jit, static_argnums=(0,))
    def _step_agents(self, state, action, max_thrust):
        """return next_state"""
        c_heading = 4
        c_dive = 5
        c_dive_throttle = 0.5
        c_throttle = 1.5
        v_min = 1.3
        g = 9.81
        v_resistance = 0.4

        if self.discrete_actions:
            action = self.action_lut[action].reshape(3).astype(jnp.float32)

        action = jnp.clip(action, -1, 1)
        u_heading, u_dive, u_throttle = action
        # x, y, z, v, heading, health, seen, alive, team = state
        alive = state["alive"]

        # Change heading and pitch based on action
        heading = state["heading"] + self.time_delta * u_heading * c_heading * alive
        # keep within pi
        heading = jnp.fmod(heading + jnp.pi, 2 * jnp.pi) - jnp.pi

        # Apply throttle
        v = state["v"] + self.time_delta * u_throttle * c_throttle * alive * max_thrust
        v = v - self.time_delta * v_resistance * jnp.square(v) * alive

        vx = v * jnp.cos(heading)
        vy = v * jnp.sin(heading)
        vz = c_dive * u_dive

        x = state["x"] + self.time_delta * vx * alive
        y = state["y"] + self.time_delta * vy * alive
        z = state["z"] + self.time_delta * vz * alive
        z = jnp.clip(z, 0, self.z_max)

        z_delta = state["z"] - z
        v_new = jnp.sqrt(
            jnp.maximum(jnp.square(v) + c_dive_throttle * g * z_delta, jnp.square(v_min))
        )  # 0 at minimum
        next_state = {
            "x": x,
            "y": y,
            "z": z,
            "v": v_new,
            "heading": heading,
            "health": state["health"],
            "alive": alive,
            "sprite": state["sprite"],
            "tracked": state["tracked"],
        }
        return next_state

    # @partial(jax.jit, static_argnums=(0,))
    @type_enforced.Enforcer
    def step(self, states: tuple[dict, dict], actions, key):
        """returns obs, next_states, reward, dones, episode_done"""
        # --- Step through individual agent physics models --- #
        v_step = jax.vmap(self._step_agents)
        agent_states, map_state = states
        agent_states = v_step(agent_states, actions, self._per_agent_thrust)

        # --- Get current agent states --- #
        # x, y, z, v, heading, health, seen, alive, teams = next_states.T
        num_agents = agent_states["x"].shape[0]
        x = agent_states["x"]
        y = agent_states["y"]
        z = agent_states["z"]
        alive = agent_states["alive"]
        teams = self._per_agent_team
        tracked = agent_states["tracked"]

        # Get agents and teams that are alive going into this step
        alive_pre = agent_states["alive"]
        alive_team1_pre = jnp.sum(alive * (teams == 0))
        alive_team2_pre = jnp.sum(alive * (teams == 1))

        # --- Check waypoint status, if enabled --- #
        hit_waypoint = 0
        waypoint_location = map_state["waypoint_location"]
        waypoint_enabled = map_state["waypoint_enabled"]
        if self.enable_waypoints:
            # Check which agents are currently in the waypoint box
            hit_waypoint = (map_state["waypoint_enabled"] > 0) * (
                (x >= waypoint_location[0])
                & (x <= waypoint_location[2])
                & (y >= waypoint_location[1])
                & (y <= waypoint_location[3])
            )

            # Generate RNG keys
            key, waypoint_time_key, waypoint_location_key, waypoint_appear_key = jax.random.split(
                key, 4
            )

            # Waypoints last for a certain period of time, this ticks the current waypoint's timer down
            waypoint_enabled = waypoint_enabled - self.time_delta

            # Waypoint disappears when the time runs up or if it is collected (`hit_waypoint` has any 1s)
            waypoint_enabled = (jnp.sum(hit_waypoint) == 0).astype(jnp.float32) * waypoint_enabled

            # If there's no waypoint active currently, then there's a 5% chance for a new one to pop up
            waypoint_appear = (waypoint_enabled <= 0) & (
                jax.random.uniform(waypoint_appear_key) < 0.05
            )

            # Sample new waypoint's location and duration
            new_waypoint_location = jax.random.uniform(
                waypoint_location_key,
                shape=(2,),
                minval=jnp.zeros(2),
                maxval=jnp.array(self.limits) - self.waypoint_size,
            )
            new_waypoint_location = jnp.array(
                [
                    new_waypoint_location[0],
                    new_waypoint_location[1],
                    new_waypoint_location[0] + self.waypoint_size,
                    new_waypoint_location[1] + self.waypoint_size,
                ]
            )
            new_waypoint_time = jax.random.uniform(waypoint_time_key, minval=5, maxval=10)

            # Refresh with new waypoint location and timer or keep the existing ones
            waypoint_location = (
                waypoint_appear * new_waypoint_location + (1 - waypoint_appear) * waypoint_location
            )
            waypoint_enabled = jnp.maximum(waypoint_enabled, 0)
            waypoint_enabled = waypoint_enabled + waypoint_appear * new_waypoint_time

        # --- Check if any agents are out of bounds in XY geographic space --- #
        out_of_bounds = (x < 0) | (x > self.limits[0]) | (y < 0) | (y > self.limits[1])
        alive = alive * (1 - out_of_bounds)

        # --- Check if any agents have collided in 3D cylindrical game space --- #
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dz = z[:, None] - z[None, :]
        collided = (
            (jnp.square(dx) + jnp.square(dy) < self.collision_range)
            & (jnp.abs(dz) < self.collision_altitude)
            & (alive[:, None] == 1)
            & (alive[None, :] == 1)
        )
        collided = collided.astype(jnp.float32)
        collided = collided - jnp.diag(jnp.diag(collided))
        collided = jnp.sum(collided, axis=1) > 0
        alive = alive * (1 - collided)

        # --- Check if any agents have collided with obstacles --- #
        # A box is an array of [x1, y1, x2, y2]
        boxes = self._maps["boxes"][map_state["map_idx"]]
        hit_box = (
            (x[:, None] > boxes[None, :, 0])
            & (x[:, None] < boxes[None, :, 2])
            & (y[:, None] > boxes[None, :, 1])
            & (y[:, None] < boxes[None, :, 3])
        )
        hit_box = jnp.sum(hit_box.astype(jnp.float32), axis=1) > 0
        hit_box = hit_box.astype(jnp.float32)
        alive = alive * (1 - hit_box)

        # --- Check which agents are within guaranteed detection range --- #
        # Very close agents are surely detected
        very_close = jnp.square(dx) + jnp.square(dy) < self.very_close_cone_depth

        # --- Check what agents see with view cone and damage cone --- #
        # Agents in cone are detected with probability inversely proportional to distance
        closeness_score = (jnp.square(dx) + jnp.square(dy)) / (
            self.cone_depth * self._per_agent_range[None, :]
        )
        closeness_score_damage = (jnp.square(dx) + jnp.square(dy)) / (
            self.damage_cone_depth * self._per_agent_damage_range[None, :]
        )

        # Calculates global bearing with respect to +x axis
        angles2 = jnp.arctan2(dy, dx)
        # Subtracting the observer's heading changes reference frame to body, relative bearing
        angles = (
            jnp.fmod(
                angles2 - agent_states["heading"][None, :] + jnp.pi + 2 * jnp.pi,
                2 * jnp.pi,
            )
            - jnp.pi
        )
        # Use half-angle for angle check because the absolute value doubles the accepted angle range
        in_cone = jnp.abs(angles) < (self.cone_angle / 2)
        in_damage_cone = jnp.abs(angles) < (self.damage_cone_angle / 2)

        # Inside view cone AND within view range (still respect depth/range)
        in_view = (closeness_score <= 1) & in_cone

        # Add stochasticity to cone detection
        if self.use_stochastic_obs:
            # Bernoulli detect with probability self.cone_detect_prob
            key, stochastic_detection_key = jax.random.split(key)
            u = jax.random.uniform(
                stochastic_detection_key, shape=in_view.shape, dtype=self.precision
            )
            # Detect iff U < p
            stochastic_detected = in_view & (u < self.cone_detect_prob)
            # Damage cone: always detect, gated by view cone
            stochastic_detected = stochastic_detected | ((closeness_score <= 1) & in_damage_cone)
        else:
            stochastic_detected = (closeness_score <= 1) & in_cone

        # --- Deal damage based on view cone and damage cone information --- #
        # Able to shoot if target is within the damage cone and within damage range
        shoot_target = (closeness_score_damage <= 1) & in_damage_cone

        # Check if agents can see each other (not same team and alive)
        can_detect = (
            (teams[:, None] != teams[None, :]) & (alive[:, None] == 1) & (alive[None, :] == 1)
        )
        # Probabilistic detection
        has_detected = can_detect & ((very_close == 1) | stochastic_detected)

        # Determine whether agent can deal damage based on detection status
        deal_damage = can_detect & shoot_target & has_detected
        deal_damage = deal_damage.astype(jnp.float32)

        # Update consecutive tracking time of opponent if it fits within damage criteria, otherwise reset tracking time
        tracked = jnp.where(deal_damage, tracked + 1.0, 0.0)

        # Damage is proportional to the number of times an agent has been tracked
        # Damage starts at 3 and increases by 1 for each time an agent is tracked up to 10
        deal_damage = (
            jnp.clip(tracked, self.min_tracking_time, self.max_tracking_time)
            - self.min_tracking_time
        )
        # Because we're following row = target, col = observer convention, sum over col
        # to get damage each target took from all the observers
        takes_damage = jnp.sum(deal_damage, axis=1)

        # --- Update agent health based on damage taken --- #
        # Check if agents are in the cone of vision of another agent
        has_detected = has_detected.astype(jnp.float32)
        seen = jnp.sum(has_detected, axis=1)  # can be greater than 1
        health = (
            agent_states["health"]
            - takes_damage * self.damage_per_second * self.time_delta
            + self.healing_per_second * self.time_delta  # automatic healing over time
            + self._per_agent_max_health * hit_waypoint  # waypoint recharges health
        ) * alive
        health = jnp.clip(health, 0, self._per_agent_max_health)  # max health is agent dependent

        # Get status of self detecting or damaging other agents, sum
        # over rows because cols represent observers
        detected_other_agent = jnp.sum(has_detected, axis=0)
        damages_other_agent = jnp.sum(deal_damage, axis=0)

        # Check if agents are dead
        alive = alive * (health > 0)

        # Check if episode is done (all agents of one team dead)
        alive_team1 = jnp.sum(alive * (teams == 0))
        alive_team2 = jnp.sum(alive * (teams == 1))

        map_state = {
            "map_idx": map_state["map_idx"],
            "waypoint_location": waypoint_location,
            "waypoint_enabled": waypoint_enabled,
            "aux_rewards_factor": map_state["aux_rewards_factor"],
            "episode_length": map_state["episode_length"] + 1,
        }
        # Both teams are dead
        episode_done = (alive_team1 == 0) & (alive_team2 == 0)
        if self.episode_ends_one_team_dead:
            # if episode_ends_one_team_dead is set then episode ends when one team is dead
            episode_done = episode_done | (alive_team1 == 0) | (alive_team2 == 0)
        if self.max_episode_length is not None:
            # if max_episode_length is not set then episode continues until all agents of one team are dead
            episode_done = episode_done | (map_state["episode_length"] >= self.max_episode_length)

        # --- Rewards --- #
        reward_info = {
            "reward_game_won": jnp.zeros((num_agents,)),
            "reward_defeat_one_opponent": jnp.zeros((num_agents,)),
            "reward_detection": jnp.zeros((num_agents,)),
            "reward_damage": jnp.zeros((num_agents,)),
            "reward_idle": jnp.zeros((num_agents,)),
            "reward_agent_disabled": jnp.zeros((num_agents,)),
            "reward_collision_agent": jnp.zeros((num_agents,)),
            "reward_collision_obstacle": jnp.zeros((num_agents,)),
        }

        # Waypoint reward
        if self.reward_hit_waypoint > 0:
            reward = self.reward_hit_waypoint * hit_waypoint
        else:
            reward = jnp.zeros(num_agents)

        # Idling reward per-agent
        reward = reward + self._per_agent_idle_reward

        # Reward for killing an opponent
        if self.reward_defeat_one_opponent > 0:
            reward_defeat_one_opponent = (
                (alive_team1 - alive_team1_pre)  #  * 0.5
                * (teams == 0)
                * alive
                / (alive_team1 + 1)
                - (alive_team2 - alive_team2_pre) * (teams == 0) * alive / (alive_team1 + 1)
                - (alive_team1 - alive_team1_pre) * (teams == 1) * alive / (alive_team2 + 1)
                + (alive_team2 - alive_team2_pre)  #  * 0.5
                * (teams == 1)
                * alive
                / (alive_team2 + 1)
            )
            reward = reward + self.reward_defeat_one_opponent * reward_defeat_one_opponent
            reward_info["reward_defeat_one_opponent"] = (
                self.reward_defeat_one_opponent * reward_defeat_one_opponent
            )

        # Detection reward
        if self.reward_detection > 0:
            # Positive reward for detecting other agents
            # Negative reward for being detected (weighted less than detecting to encourage exploration)
            reward_detection = (
                0.5 * detected_other_agent * self.time_delta * alive
                - 0.2 * seen * self.time_delta * alive
            )
            reward = reward + self.reward_detection * reward_detection
            reward_info["reward_detection"] = self.reward_detection * reward_detection

        # Damage reward
        if self.reward_damage > 0:
            # Positive reward for dealing damage to other agents
            # Negative reward for taking damage from other agents
            reward_damage = (
                damages_other_agent * self.time_delta * alive
                - 0.5 * takes_damage * self.time_delta * alive
            )
            reward = reward + self.reward_damage * reward_damage
            reward_info["reward_damage"] = self.reward_damage * reward_damage

        # Negative reward for being idle
        if self.reward_idle > 0:
            reward_idle = 0.01 * self.time_delta * alive
            reward = reward - self.reward_idle * reward_idle
            reward_info["reward_idle"] = -self.reward_idle * reward_idle

        # Negative reward for collisions, going out of bounds and hitting boxes
        # Negative reward for dying (health drops to 0)
        if self.reward_collision_agent > 0:
            reward_collision_agent = collided
            reward = reward - self.reward_collision_agent * reward_collision_agent
            reward_info["reward_collision_agent"] = (
                -self.reward_collision_agent * reward_collision_agent
            )

        if self.reward_collision_obstacle > 0:
            reward_collision_obstacle = (out_of_bounds + hit_box) * alive
            reward = reward - self.reward_collision_obstacle * reward_collision_obstacle
            reward_info["reward_collision_obstacle"] = (
                -self.reward_collision_obstacle * reward_collision_obstacle
            )

        if self.reward_agent_disabled > 0:
            reward_agent_disabled = (1 - alive) * alive_pre
            reward = reward - self.reward_agent_disabled * reward_agent_disabled
            reward_info["reward_agent_disabled"] = (
                -self.reward_agent_disabled * reward_agent_disabled
            )

        # Positive reward for winning the game (weighted by number of agents alive)
        game_won_reward = (alive_team2 == 0) * (teams == 0) * alive + (alive_team1 == 0) * (
            teams == 1
        ) * alive
        # Enable curriculum learning by scaling the auxiliary reward
        # if aux factor is set to zero only the end of the winning of the game will give a reward
        reward = map_state["aux_rewards_factor"] * reward + self.reward_game_won * game_won_reward
        reward_info["reward_game_won"] = self.reward_game_won * game_won_reward

        # Normalize reward to number of agents. Make the reward in a similar level for different number of agents
        if self.divide_reward_by_team_size:
            reward = reward / num_agents
        reward = reward * alive_pre  # if agent was already dead, reward is zero

        agent_states = {
            "x": x,
            "y": y,
            "z": z,
            "v": agent_states["v"],
            "heading": agent_states["heading"],
            "health": health,
            "alive": alive,
            "sprite": agent_states["sprite"],
            "tracked": tracked,
        }
        if self.debug_reward:
            for k, v in reward_info.items():
                agent_states[k] = v

        # --- Set up next state --- #
        next_states = (agent_states, map_state)

        # --- Get agent observations from state --- #
        v_get_observation = jax.vmap(self.get_observation, in_axes=(None, None, None, 0, 0))
        key = jax.random.split(key, num_agents)
        obs = v_get_observation(
            next_states, has_detected, takes_damage > 0, key, jnp.arange(num_agents)
        )

        # --- Set up agent done status --- #
        dones = (1 - alive).astype(jnp.bool_)

        return obs, next_states, reward, dones, episode_done

    def get_dones(self, states):
        return states[0]["alive"]

    # @partial(jax.jit, static_argnums=(0,))
    @type_enforced.Enforcer
    def get_observation(self, states: tuple[dict, dict], has_detected, took_damage, rng, agent_id):
        """
        if self._obs_type == "rgb":
            return rgb_obs
        elif self._obs_type == "vector":
            return vector_obs
        elif self._obs_type == "rgb_vector":
            return rgb_obs, vector_obs
        """
        agent_states, map_state = states
        num_agents = agent_states["x"].shape[0]
        x = agent_states["x"]
        y = agent_states["y"]
        z = agent_states["z"]
        alive = agent_states["alive"]
        teams = self._per_agent_team
        heading = agent_states["heading"]

        # --- Determine comms --- #
        if self.use_stochastic_comm:
            # Get distance from current agent to other agents
            distance = jnp.sqrt(
                jnp.square(x[agent_id, None] - x[None, :])
                + jnp.square(y[agent_id, None] - y[None, :])
            )

            # Scale distance by max communication range, can be greater than 1 which is clipped
            distance = distance / self.max_communication_range
            distance = jnp.clip(distance, 0, 1)

            # Dead and enemy agents are out of communication range (their distance is always > rand)
            distance = distance + (1 - alive[None, :]) * 2 + (teams[agent_id] != teams[None, :])
            rand = jax.random.uniform(rng, shape=distance.shape, dtype=self.precision)
            communicate = distance <= rand

            # Only pass through detections from allies that you can communicate with
            seen = (
                has_detected + jnp.eye(has_detected.shape[0], dtype=self.precision)
            ) * communicate
        else:
            # Only pass through detections from allies that you can communicate with
            seen = has_detected + jnp.eye(has_detected.shape[0], dtype=self.precision) * (
                teams[agent_id] == teams[None, :]
            ) * (alive[None, :] > 0)

        seen = jnp.sum(seen, axis=1) > 0

        # --- RGB observations --- #
        rgb_obs, vector_obs = None, None
        if "rgb" in self._obs_type:
            x = jnp.round(x * self.resolution[0] / self.limits[0]).astype(jnp.int32)
            y = jnp.round(y * self.resolution[1] / self.limits[1]).astype(jnp.int32)
            z = jnp.round(z * 255 / self.z_max).astype(jnp.uint8)
            # alive = alive.astype(jnp.uint8)
            teams = teams.astype(jnp.uint8)
            rgb_obs = self._prerendered_maps[map_state["map_idx"]]

            # Border is red
            rgb_obs = rgb_obs.at[:, 0, 0].max(255)
            rgb_obs = rgb_obs.at[0, :, 0].max(255)
            rgb_obs = rgb_obs.at[self.resolution[0] - 1, :, 0].max(255)
            # Border is white if agent is not taken damage
            not_taken_damage = (1 - took_damage[agent_id]).astype(jnp.uint8)
            rgb_obs = rgb_obs.at[:, 0, :].max(255 * not_taken_damage)
            rgb_obs = rgb_obs.at[0, :, :].max(255 * not_taken_damage)
            rgb_obs = rgb_obs.at[self.resolution[0] - 1, :, :].max(255 * not_taken_damage)

            if self.enable_waypoints:
                waypoint_num_pixels_x = max(
                    int(self.waypoint_size * self.resolution[0] / self.limits[0]), 1
                )
                waypoint_num_pixels_y = max(
                    int(self.waypoint_size * self.resolution[1] / self.limits[1]), 1
                )
                waypoint_start_x = jnp.round(
                    map_state["waypoint_location"][0] * self.resolution[0] / self.limits[0]
                ).astype(jnp.int32)
                waypoint_start_y = jnp.round(
                    map_state["waypoint_location"][1] * self.resolution[1] / self.limits[1]
                ).astype(jnp.int32)
                waypoint_color = (
                    (map_state["waypoint_enabled"] > 0) * jnp.array([127, 0, 127])
                ).astype(jnp.uint8)
                for ix in range(waypoint_num_pixels_x):
                    for iy in range(waypoint_num_pixels_y):
                        rgb_obs = rgb_obs.at[waypoint_start_x + ix, waypoint_start_y + iy].set(
                            waypoint_color
                        )

            # Health bar is green
            rgb_obs = rgb_obs.at[:, self.resolution[1] - 1, 1].max(255)
            health_cutoff = jnp.int32(
                self.resolution[1]
                * agent_states["health"][agent_id]
                / self._per_agent_max_health[agent_id]
            )
            rb_channel = jnp.where(jnp.arange(self.resolution[1]) < health_cutoff, 0, 255)
            rgb_obs = rgb_obs.at[:, self.resolution[1] - 1, 0].set(rb_channel)
            rgb_obs = rgb_obs.at[:, self.resolution[1] - 1, 2].set(rb_channel)
            # rgb_obs = rgb_obs.at[:, self.resolution[1] - 1, 0].max(255)

            tail_length = 5
            own_team = (teams == teams[agent_id]).astype(jnp.uint8)

            # Draw tails first
            for i in range(1, tail_length):
                intensity = jnp.uint8(255 - 255 * i / tail_length)
                xi = x - i * jnp.cos(heading)
                yi = y - i * jnp.sin(heading)
                xi = jnp.clip(xi, 0, self.resolution[0] - 1).astype(jnp.int32)
                yi = jnp.clip(yi, 0, self.resolution[1] - 1).astype(jnp.int32)
                # Own team tail is blue
                rgb_obs = rgb_obs.at[xi, yi, 2].add(intensity * own_team * seen, mode="drop")
                # Other team tail is red
                rgb_obs = rgb_obs.at[xi, yi, 0].add(intensity * (1 - own_team) * seen, mode="drop")
                # Ego tail is white
                rgb_obs = rgb_obs.at[xi[agent_id], yi[agent_id], :].max(intensity)

            # Draw agents
            rgb_obs = draw_agents_from_ego(
                rgb_obs, x, y, z, teams, seen, agent_id, agent_states["sprite"]
            )
            # rgb_obs = jnp.maximum(
            #     rgb_obs, 255 * (1 - alive[agent_id].astype(jnp.uint8))
            # )

        # --- Vector observations --- #
        if "vector" in self._obs_type:
            # sort by distance from ego
            distance = jnp.sqrt(jnp.square(x[agent_id] - x) + jnp.square(y[agent_id] - y))
            # agents that are not seen are at the end of the list
            ranking = jnp.where(seen, distance, jnp.square(self.limits[0] + self.limits[1]) + 1)
            # For some reason we need to manually set the ego agent to location 0
            ranking = jnp.where(jnp.arange(num_agents) == agent_id, -1, ranking)

            sorted_indices = jnp.argsort(ranking).flatten()
            relative_team = (teams == teams[agent_id]).astype(jnp.float32)
            relative_x = agent_states["x"] - agent_states["x"][agent_id]
            relative_y = agent_states["y"] - agent_states["y"][agent_id]
            relative_z = agent_states["z"] - agent_states["z"][agent_id]
            relative_heading = agent_states["heading"] - agent_states["heading"][agent_id]
            relative_v = agent_states["v"] - agent_states["v"][agent_id]
            relative_heading = jnp.fmod(relative_heading + jnp.pi, 2 * jnp.pi) - jnp.pi

            # Overwrite the first element with the absolute value of the ego agent
            relative_x = relative_x.at[agent_id].set(agent_states["x"][agent_id])
            relative_y = relative_y.at[agent_id].set(agent_states["y"][agent_id])
            relative_z = relative_z.at[agent_id].set(agent_states["z"][agent_id])
            relative_heading = relative_heading.at[agent_id].set(agent_states["heading"][agent_id])
            relative_v = relative_v.at[agent_id].set(agent_states["v"][agent_id])

            vector_obs = jnp.stack(
                [
                    # ego obs
                    seen[sorted_indices][: self.max_agent_in_vec_obs],
                    (seen * relative_team)[sorted_indices][: self.max_agent_in_vec_obs],
                    (seen * relative_x)[sorted_indices][: self.max_agent_in_vec_obs]
                    / self.limits[0],
                    (seen * relative_y)[sorted_indices][: self.max_agent_in_vec_obs]
                    / self.limits[1],
                    (seen * relative_z)[sorted_indices][: self.max_agent_in_vec_obs] / self.z_max,
                    (seen * relative_heading)[sorted_indices][: self.max_agent_in_vec_obs],
                    (seen * relative_v)[sorted_indices][: self.max_agent_in_vec_obs]
                    / jnp.sqrt(jnp.square(self.limits[0]) + jnp.square(self.limits[1])),
                    # normalize by diagonal of the map
                    (seen * agent_states["health"])[sorted_indices][: self.max_agent_in_vec_obs],
                    (seen * agent_states["sprite"])[sorted_indices][
                        : self.max_agent_in_vec_obs
                    ],  # type of agent
                ],
                axis=1,  # stack along the second dimension
            )
            vector_obs = vector_obs.flatten()

            additional_obs = []
            if self.enable_waypoints:
                # normalize
                wp_normalizer_sub = jnp.array(
                    [
                        agent_states["x"][agent_id],
                        agent_states["y"][agent_id],
                        agent_states["x"][agent_id],
                        agent_states["y"][agent_id],
                    ]
                )
                wp_normalizer_div = jnp.array(
                    [
                        self.limits[0],
                        self.limits[1],
                        self.limits[0],
                        self.limits[1],
                    ]
                )

                relative_waypoint_loc = (
                    map_state["waypoint_location"] - wp_normalizer_sub
                ) / wp_normalizer_div

                additional_obs.append(relative_waypoint_loc.flatten())
                additional_obs.append(jnp.array([map_state["waypoint_enabled"]]))
            if self._maps_names != "empty":
                boxes = self._maps["boxes"][map_state["map_idx"]]
                box_normalizer_div = jnp.array(
                    [
                        [
                            self.limits[0],
                            self.limits[1],
                            self.limits[0],
                            self.limits[1],
                        ]
                    ]
                )
                boxes = boxes / jnp.expand_dims(box_normalizer_div, axis=0)
                additional_obs.append(boxes.flatten())
            if len(additional_obs) > 0:
                vector_obs = jnp.concatenate([vector_obs] + additional_obs)
                # append obstacles and waypoints to the end of the vector
            # now obs of ego is at the beginning of the vector
            # obs other agents are consecutive and sorted by distance from ego

        if self._obs_type == "rgb":
            return rgb_obs
        elif self._obs_type == "vector":
            return vector_obs
        elif self._obs_type == "rgb_vector":
            return rgb_obs, vector_obs
        else:
            raise ValueError(
                f"Unknown observation type {self._obs_type}! This code should not be reached"
            )

    @partial(jax.jit, static_argnums=(0,))
    @type_enforced.Enforcer
    def get_global_observation(self, states: tuple[dict, dict]):
        """return obs with FOV cones overlaid by team color"""
        agent_states, map_state = states
        x = agent_states["x"]
        y = agent_states["y"]
        z = agent_states["z"]
        alive = agent_states["alive"]
        teams = self._per_agent_team
        heading = agent_states["heading"]

        # --- Convert to pixel-space for drawing ---
        x_pix = jnp.round(x * self.resolution[0] / self.limits[0]).astype(jnp.int32)
        y_pix = jnp.round(y * self.resolution[1] / self.limits[1]).astype(jnp.int32)
        z_u8 = jnp.round(z * 255 / self.z_max).astype(jnp.uint8)
        alive_u8 = alive.astype(jnp.uint8)
        teams_u8 = teams.astype(jnp.uint8)
        obs = self._prerendered_maps[map_state["map_idx"]]

        # --- Draw borders --- #
        obs = obs.at[:, 0, :].max(255)
        obs = obs.at[:, self.resolution[1] - 1, :].max(255)
        obs = obs.at[0, :, :].max(255)
        obs = obs.at[self.resolution[0] - 1, :, :].max(255)

        # --- Draw waypoints --- #
        if self.enable_waypoints:
            waypoint_num_pixels_x = max(
                int(self.waypoint_size * self.resolution[0] / self.limits[0]), 1
            )
            waypoint_num_pixels_y = max(
                int(self.waypoint_size * self.resolution[1] / self.limits[1]), 1
            )
            waypoint_start_x = jnp.round(
                map_state["waypoint_location"][0] * self.resolution[0] / self.limits[0]
            ).astype(jnp.int32)
            waypoint_start_y = jnp.round(
                map_state["waypoint_location"][1] * self.resolution[1] / self.limits[1]
            ).astype(jnp.int32)
            waypoint_color = (
                (map_state["waypoint_enabled"] > 0) * jnp.array([127, 0, 127])
            ).astype(jnp.uint8)
            for ix in range(waypoint_num_pixels_x):
                for iy in range(waypoint_num_pixels_y):
                    obs = obs.at[waypoint_start_x + ix, waypoint_start_y + iy].set(waypoint_color)

        # --- Draw tails --- #
        team1 = teams_u8
        team2 = 1 - teams_u8
        tail_length = 5
        for i in range(1, tail_length):
            intensity = jnp.uint8(255 - 255 * i / tail_length)
            xi = x_pix - i * jnp.cos(heading)
            yi = y_pix - i * jnp.sin(heading)
            xi = jnp.clip(xi, 0, self.resolution[0] - 1).astype(jnp.int32)
            yi = jnp.clip(yi, 0, self.resolution[1] - 1).astype(jnp.int32)
            obs = obs.at[xi, yi, 0].add(intensity * team1 * alive_u8, mode="drop")
            obs = obs.at[xi, yi, 2].add(intensity * team2 * alive_u8, mode="drop")

        # --- Draw agents --- #
        obs = draw_all_agents(obs, x_pix, y_pix, z_u8, teams_u8, alive_u8, agent_states["sprite"])

        # --- View cone overlay (not damage cone) --- #
        if self.enable_view_cone_overlay:
            # Pixel grid (X first dimension = height, Y second = width)
            gx = jnp.arange(self.resolution[0], dtype=self.precision)[None, :, None]  # [1, H, 1]
            gy = jnp.arange(self.resolution[1], dtype=self.precision)[None, None, :]  # [1, 1, W]

            # Agents broadcast
            xi = x_pix.astype(self.precision)[:, None, None]  # [N, 1, 1]
            yi = y_pix.astype(self.precision)[:, None, None]  # [N, 1, 1]
            hi = heading[:, None, None]  # [N, 1, 1]
            alive_mask = alive[:, None, None] > 0  # [N, 1, 1]
            team1_mask = teams_u8[:, None, None]  # [N, 1, 1]
            team2_mask = (1 - teams_u8)[:, None, None]  # [N, 1, 1]

            # Pixel deltas in PIXELS -> broadcast to [N, H, W]
            dx_pix = jnp.broadcast_to(
                gx - xi, (x_pix.shape[0], self.resolution[0], self.resolution[1])
            )
            dy_pix = jnp.broadcast_to(
                gy - yi, (y_pix.shape[0], self.resolution[0], self.resolution[1])
            )

            # Convert to ENV units before distance/angle checks (preserves cone size)
            sx = self.limits[0] / self.resolution[0]
            sy = self.limits[1] / self.resolution[1]
            dx_env = dx_pix * sx
            dy_env = dy_pix * sy

            dist2 = dx_env * dx_env + dy_env * dy_env
            dirx = jnp.cos(hi)
            diry = jnp.sin(hi)
            dot = dx_env * dirx + dy_env * diry

            # Angle check via cosine threshold (use HALF-FOV)
            half = (
                0.5 * self.cone_angle
            )  # radians; if using degrees, do: 0.5 * jnp.deg2rad(self.cone_angle)
            cosang = dot / jnp.sqrt(jnp.maximum(dist2, 1e-6))
            cos_thresh = jnp.cos(half)
            ang_ok = cosang >= cos_thresh

            # Front-gate only when FoV <= 180° (half <= 90°)
            # Equivalent to: front_ok = jnp.where(half <= jnp.pi/2, dot > 0, True)
            front_ok = (dot > 0) | (half > (jnp.pi / 2))

            # Per-agent range in ENV units
            # cone_depth is squared; effective range^2 = cone_depth * per_agent_range
            range2_env = (self.cone_depth * self._per_agent_range).astype(self.precision)[
                :, None, None
            ]
            in_range = dist2 <= range2_env

            fov_mask = (ang_ok & front_ok & in_range & alive_mask).astype(jnp.uint16)  # [N, H, W]

            # Accumulate by team and overlay with modest opacity
            opacity = jnp.uint16(64)  # tweak 32..96 for lighter/darker cones
            r_overlay = jnp.clip(
                jnp.sum(fov_mask * team1_mask.astype(jnp.uint16), axis=0) * opacity, 0, 255
            ).astype(jnp.uint8)
            b_overlay = jnp.clip(
                jnp.sum(fov_mask * team2_mask.astype(jnp.uint16), axis=0) * opacity, 0, 255
            ).astype(jnp.uint8)

            # Blend by taking the max so cones are visible but don't remove existing features
            obs = obs.at[:, :, 0].max(r_overlay)
            obs = obs.at[:, :, 2].max(b_overlay)

        # Always return the image (even if overlay disabled)
        return obs

    @partial(jax.jit, static_argnums=(0,))
    @type_enforced.Enforcer
    def set_aux_reward_factor(self, state: tuple[dict, dict], aux_rewards_factor):
        agent_states, map_states = state
        map_states["aux_rewards_factor"] = aux_rewards_factor
        return (agent_states, map_states)

    # @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng):
        """return obs, state"""
        rng = jax.random.split(rng, 7)

        map_idx = jax.random.randint(rng[5], shape=(), minval=0, maxval=len(self._maps))

        map_state = {
            "map_idx": map_idx,
            "waypoint_location": jnp.zeros(4),
            "waypoint_enabled": jnp.float32(0),
            "aux_rewards_factor": jnp.float32(1),
            "episode_length": jnp.int32(0),
        }
        x = self._per_agent_team * jax.random.uniform(
            rng[0],
            shape=(self.n_agents,),
            minval=self._maps["start_pos_team_a"][map_idx][0],
            maxval=self._maps["start_pos_team_a"][map_idx][2],
            dtype=self.precision,
        ) + (1 - self._per_agent_team) * jax.random.uniform(
            rng[0],
            shape=(self.n_agents,),
            minval=self._maps["start_pos_team_b"][map_idx][0],
            maxval=self._maps["start_pos_team_b"][map_idx][2],
            dtype=self.precision,
        )
        y = self._per_agent_team * jax.random.uniform(
            rng[1],
            shape=(self.n_agents,),
            minval=self._maps["start_pos_team_a"][map_idx][1],
            maxval=self._maps["start_pos_team_a"][map_idx][3],
            dtype=self.precision,
        ) + (1 - self._per_agent_team) * jax.random.uniform(
            rng[1],
            shape=(self.n_agents,),
            minval=self._maps["start_pos_team_b"][map_idx][1],
            maxval=self._maps["start_pos_team_b"][map_idx][3],
            dtype=self.precision,
        )
        z = jax.random.uniform(
            rng[2],
            shape=(self.n_agents,),
            minval=self._maps["start_height"][map_idx][0],
            maxval=self._maps["start_height"][map_idx][1],
            dtype=self.precision,
        )
        v = jnp.ones(self.n_agents, dtype=self.precision)
        heading = self._per_agent_team * jax.random.uniform(
            rng[4],
            shape=(self.n_agents,),
            minval=self._maps["start_heading_a"][map_idx][0],
            maxval=self._maps["start_heading_a"][map_idx][1],
            dtype=self.precision,
        ) + (1 - self._per_agent_team) * jax.random.uniform(
            rng[4],
            shape=(self.n_agents,),
            minval=self._maps["start_heading_b"][map_idx][0],
            maxval=self._maps["start_heading_b"][map_idx][1],
            dtype=self.precision,
        )
        # To avoid collisions in the reset step, we resample agents that are too close to each other or to boxes
        # up to 3 times
        for i in range(3):
            # slightly enlarge collision radius to avoid collisions in the first steps
            collision_enlarge = 0.1

            # Resample agents if there is a collision (with other agents or boxes)
            rng = jax.random.split(rng[-1], 4)
            collided = (
                jnp.square(x[:, None] - x[None, :]) + jnp.square(y[:, None] - y[None, :])
                < self.collision_range + collision_enlarge
            ) & (jnp.abs(z[:, None] - z[None, :]) < self.collision_altitude + collision_enlarge)
            collided = collided.astype(jnp.float32)
            collided = collided - jnp.diag(jnp.diag(collided))
            collided = jnp.sum(collided, axis=1) > 0

            boxes = self._maps["boxes"][map_state["map_idx"]]
            hit_box = (
                (x[:, None] > boxes[None, :, 0] - collision_enlarge)
                & (x[:, None] < boxes[None, :, 2] + collision_enlarge)
                & (y[:, None] > boxes[None, :, 1] - collision_enlarge)
                & (y[:, None] < boxes[None, :, 3] + collision_enlarge)
            )
            hit_box = jnp.sum(hit_box.astype(jnp.float32), axis=1) > 0
            need_to_resample = collided | hit_box
            x_new = self._per_agent_team * jax.random.uniform(
                rng[0],
                shape=(self.n_agents,),
                minval=self._maps["start_pos_team_a"][map_idx][0],
                maxval=self._maps["start_pos_team_a"][map_idx][2],
                dtype=self.precision,
            ) + (1 - self._per_agent_team) * jax.random.uniform(
                rng[0],
                shape=(self.n_agents,),
                minval=self._maps["start_pos_team_b"][map_idx][0],
                maxval=self._maps["start_pos_team_b"][map_idx][2],
                dtype=self.precision,
            )
            y_new = self._per_agent_team * jax.random.uniform(
                rng[1],
                shape=(self.n_agents,),
                minval=self._maps["start_pos_team_a"][map_idx][1],
                maxval=self._maps["start_pos_team_a"][map_idx][3],
                dtype=self.precision,
            ) + (1 - self._per_agent_team) * jax.random.uniform(
                rng[1],
                shape=(self.n_agents,),
                minval=self._maps["start_pos_team_b"][map_idx][1],
                maxval=self._maps["start_pos_team_b"][map_idx][3],
                dtype=self.precision,
            )
            z_new = jax.random.uniform(
                rng[2],
                shape=(self.n_agents,),
                minval=self._maps["start_height"][map_idx][0],
                maxval=self._maps["start_height"][map_idx][1],
                dtype=self.precision,
            )
            x = jnp.where(need_to_resample, x_new, x)
            y = jnp.where(need_to_resample, y_new, y)
            z = jnp.where(need_to_resample, z_new, z)

        health = jnp.ones((self.n_agents,), dtype=jnp.float32)
        alive = jnp.ones((self.n_agents,), dtype=jnp.float32)
        tracked = jnp.zeros((self.n_agents, self.n_agents), dtype=jnp.float32)
        # TODO: Some values are static, and does not need to be in the state
        agent_state = {
            "x": x,
            "y": y,
            "z": z,
            "v": v,
            "heading": heading,
            "health": health,
            "alive": alive,
            "sprite": self._per_agent_sprites,
            "tracked": tracked,
        }
        if self.debug_reward:
            agent_state["reward_game_won"] = jnp.zeros((self.n_agents,), dtype=self.precision)
            agent_state["reward_defeat_one_opponent"] = jnp.zeros(
                (self.n_agents,), dtype=self.precision
            )
            agent_state["reward_detection"] = jnp.zeros((self.n_agents,), dtype=self.precision)
            agent_state["reward_damage"] = jnp.zeros((self.n_agents,), dtype=self.precision)
            agent_state["reward_idle"] = jnp.zeros((self.n_agents,), dtype=self.precision)
            agent_state["reward_agent_disabled"] = jnp.zeros((self.n_agents,), dtype=self.precision)
            agent_state["reward_collision_agent"] = jnp.zeros(
                (self.n_agents,), dtype=self.precision
            )
            agent_state["reward_collision_obstacle"] = jnp.zeros(
                (self.n_agents,), dtype=self.precision
            )

        state = (agent_state, map_state)

        v_get_observation = jax.vmap(self.get_observation, in_axes=(None, None, None, 0, 0))
        rng = jax.random.split(rng[-1], x.shape[0])
        obs = v_get_observation(
            state,
            jnp.zeros((self.n_agents, self.n_agents), dtype=self.precision),
            jnp.zeros(self.n_agents, dtype=self.precision),
            rng,
            jnp.arange(x.shape[0]),
        )

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    @type_enforced.Enforcer
    def reset_done_episodes(self, obs, states: tuple[dict, dict], ep_dones, rng_key):
        """return obs, reset_states"""
        batch_size = ep_dones.shape[0]
        rng_key = jax.random.split(rng_key, batch_size)
        new_obs, new_states = self.v_reset(rng_key)

        reset_states = jax.tree_util.tree_map(
            lambda x, y: jnp.where(
                # Create a tuple of (batch_size, 1, ... 1) to broadcast
                ep_dones.reshape(*([batch_size] + [1] * (x.ndim - 1))),
                x,
                y,
            ),
            new_states,
            states,
        )
        if self._obs_type == "vector":
            obs = jnp.where(ep_dones.reshape(batch_size, 1, 1), new_obs, obs)
        elif self._obs_type == "rgb":
            obs = jnp.where(ep_dones.reshape(batch_size, 1, 1, 1, 1), new_obs, obs)
        else:
            # "vector_rgb" -> obs is a tuple of (obs0, obs1)
            obs0 = jnp.where(ep_dones.reshape(batch_size, 1, 1, 1, 1), new_obs[0], obs[0])
            obs1 = jnp.where(ep_dones.reshape(batch_size, 1, 1), new_obs[1], obs[1])
            obs = (obs0, obs1)
        return obs, reset_states

    @classmethod
    def get_initial_state(
        cls,
        x=0,
        y=0,
        z=0,
        v=1,
        heading=0,
        health=1,
        alive=1,
        sprite=1,
    ):
        return {
            "x": x,
            "y": y,
            "z": z,
            "v": v,
            "heading": heading,
            "health": health,
            "alive": alive,
            "sprite": sprite,
        }

    @classmethod
    def action(cls, heading=0, dive=0, speed=0):
        return jnp.array([heading, dive, speed])


class EnvFrameStack:
    """
    Stack the observation frames along the height
    Only support RGB observation, since the vector observations have speed information
    """

    def __init__(self, env, nstack):
        self.env = env
        self.nstack = nstack

        obs_space = env.observation_space  # wrapped ob space
        self.shape_dim_last = obs_space.shape[0]
        if len(obs_space.shape) == 3:
            low = jnp.repeat(obs_space.low, self.nstack, axis=0)
            high = jnp.repeat(obs_space.high, self.nstack, axis=0)
        else:
            raise NotImplementedError("Stack env only support RGB observation")

        self.stacked_obs = None

        self.observation_space = Box(low=low, high=high)

    def v_step(self, states, actions, key):
        """return self.stacked_obs, states, r, d, ep_dones"""
        obs, states, r, d, ep_dones = self.env.v_step(states, actions, key)

        if self.stacked_obs is None:
            self.stacked_obs = jnp.zeros(
                (
                    obs.shape[0],
                    obs.shape[1],
                    self.nstack * obs.shape[2],
                    *obs.shape[3:],
                ),
                dtype=obs.dtype,
            )

        self.stacked_obs = self.stacked_obs.at[:, :, : -self.shape_dim_last, :, :].set(
            self.stacked_obs[:, :, self.shape_dim_last :, :, :]
        )

        self.obs = obs

        self.stacked_obs = self.stacked_obs.at[:, :, -self.shape_dim_last :, :, :].set(obs)

        return self.stacked_obs, states, r, d, ep_dones

    @type_enforced.Enforcer
    def reset_done_episodes(self, obs, states: tuple[dict, dict], ep_dones, key):
        """return self.stacked_obs, states"""
        _, states = self.env.reset_done_episodes(self.obs, states, ep_dones, key)
        batch_size = self.stacked_obs.shape[0]
        new_obs = jnp.zeros_like(self.stacked_obs)
        self.stacked_obs = jnp.where(
            ep_dones.reshape(batch_size, 1, 1, 1, 1), new_obs, self.stacked_obs
        )

        return self.stacked_obs, states

    def v_reset(self, key):
        """return obs, states"""
        obs, states = self.env.v_reset(key)
        self.obs = obs

        return obs, states

    def reset(self):
        """return self.stacked_obs"""
        obs = self.env.reset()
        self.stacked_obs[:, -self.shape_dim_last :] = obs
        self.obs = obs
        return self.stacked_obs
