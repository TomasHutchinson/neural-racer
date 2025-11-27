import numpy as np
import track

class CarManager:
    def __init__(
        self,
        spline_points,
        left_edge,
        right_edge,
        N_players=1,
        N_ai=0,
        car_width=40,
        car_height=20,
        max_speed=600.0,
        acceleration=500.0,
        brake_strength=2.0,
        turn_speed=180.0,
        bounce_factor=0.5,
        N_rays=9,
        ray_length=30.0,
        ray_fan_deg=130.0,
    ):
        self.spline_points = np.asarray(spline_points)
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.N_players = N_players
        self.N_ai = N_ai
        self.N = N_players + N_ai
        self.car_width = car_width
        self.car_height = car_height
        self.radius = max(car_width, car_height) / 2.0
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.brake_strength = brake_strength
        self.turn_speed = turn_speed
        self.bounce_factor = bounce_factor
        self.N_rays = N_rays
        self.ray_length = ray_length
        self.ray_fan_deg = ray_fan_deg
        self.reset_self()

    def reset_self(self):
        M = len(self.spline_points)
        start_indices = np.linspace(0, M - M, self.N, dtype=int)
        self.car_pos = self.spline_points[start_indices, :2].astype(float) + np.random.randn(self.N, 2) * 2.0
        self.car_angle = self.spline_points[start_indices, 2].astype(float).copy()
        self.car_velocity = np.zeros((self.N, 2), dtype=float)
        self.ai_target_idx = start_indices.copy()
        self.cumulative_distance = np.zeros(self.N, dtype=float)
        self._last_pos = self.car_pos.copy()

    @staticmethod
    def apply_friction_vec(vel, dt, factor=0.1):
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        friction = np.minimum(factor * speed, 1.0)
        return vel - vel * friction * dt

    @staticmethod
    def clamp_speed(vel, max_speed):
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        mask = speed > max_speed
        if np.any(mask):
            vel[mask.flatten()] = vel[mask.flatten()] / speed[mask] * max_speed
        return vel

    def reflect_velocity_vec(self, vels, normals):
        dot = np.sum(vels * normals, axis=1, keepdims=True)
        mask = (dot < 0).flatten()
        if not np.any(mask):
            return vels
        reflected = vels - 2 * dot * normals
        vels[mask] = reflected[mask] * self.bounce_factor
        return vels

    def apply_player_controls(self, keys, dt):
        fwd = np.array([np.cos(np.deg2rad(self.car_angle[0])),
                        np.sin(np.deg2rad(self.car_angle[0]))])
        if keys["up"]:
            self.car_velocity[0] += fwd * self.acceleration * dt
        if keys["down"]:
            self.car_velocity[0] -= fwd * self.brake_strength * dt
        if keys["left"]:
            self.car_angle[0] -= self.turn_speed * dt
        if keys["right"]:
            self.car_angle[0] += self.turn_speed * dt
        self.car_angle %= 360.0

    def update(self, dt):
        self.car_velocity = self.clamp_speed(self.car_velocity, self.max_speed)
        self.car_velocity = self.apply_friction_vec(self.car_velocity, dt)
        self.car_pos += self.car_velocity * dt
        dpos = self.car_pos - self._last_pos
        self.cumulative_distance += np.linalg.norm(dpos, axis=1)
        self._last_pos = self.car_pos.copy()
        collision_found, normals, penetrations, _ = track.vectorized_collision(
            self.car_pos, self.radius, self.left_edge, self.right_edge
        )
        if np.any(collision_found):
            mask = collision_found
            self.car_pos[mask] += normals[mask] * penetrations[mask][:, None]
            self.car_velocity[mask] = self.reflect_velocity_vec(
                self.car_velocity[mask], normals[mask]
            )
            self.car_velocity = self.apply_friction_vec(self.car_velocity, dt)
            speeds = np.linalg.norm(self.car_velocity, axis=1)
            self.car_velocity[speeds < 0.01] = 0.0
        return collision_found

    def compute_rays(self):
        return track.compute_rays(
            self.car_pos,
            self.car_angle,
            self.N_rays,
            self.ray_length,
            self.ray_fan_deg,
        )

    def raycast(self, rays_start, rays_end):
        return track.vectorized_ray_cast(
            rays_start,
            rays_end,
            self.left_edge,
            self.right_edge
        )
