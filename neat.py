import numpy as np
import pygame

class SimpleNN:
    def __init__(self, n_input, n_hidden1=16, n_hidden2=16, n_output=2, N_net=1):
        self.N_net = N_net
        self.W1 = np.random.randn(N_net, n_hidden1, n_input) * np.sqrt(2/n_input)
        self.b1 = np.zeros((N_net, n_hidden1))
        self.W2 = np.random.randn(N_net, n_hidden2, n_hidden1) * np.sqrt(2/n_hidden1)
        self.b2 = np.zeros((N_net, n_hidden2))
        self.W3 = np.random.randn(N_net, n_output, n_hidden2) * np.sqrt(2/n_hidden2)
        self.b3 = np.zeros((N_net, n_output))

    def forward(self, X):
        self.X = X
        h1 = np.tanh(np.einsum('nij,nj->ni', self.W1, X) + self.b1)
        h2 = np.tanh(np.einsum('nij,nj->ni', self.W2, h1) + self.b2)
        out = np.tanh(np.einsum('nij,nj->ni', self.W3, h2) + self.b3)
        return out

    def copy(self):
        nn = SimpleNN.__new__(SimpleNN)
        nn.N_net = self.N_net
        nn.W1, nn.b1 = self.W1.copy(), self.b1.copy()
        nn.W2, nn.b2 = self.W2.copy(), self.b2.copy()
        nn.W3, nn.b3 = self.W3.copy(), self.b3.copy()
        return nn

    def mutate(self, rate=0.1, scale=0.3):
        for param in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]:
            mutation = (np.random.rand(*param.shape) < rate) * np.random.randn(*param.shape) * scale
            param += mutation

class AIManager:
    def __init__(self, car_manager, N_ai=20, n_hidden1=16, n_hidden2=16):
        self.cm = car_manager
        self.N_ai = N_ai
        self.ai_indices = np.arange(self.cm.N_players, self.cm.N_players + N_ai)
        n_input = self.cm.N_rays + 4
        n_output = 4
        self.nets = SimpleNN(n_input, n_hidden1, n_hidden2, n_output, N_net=N_ai)
        self.fitness = np.zeros(N_ai)
        self.last_target_idx = np.zeros(N_ai, dtype=int)
        self.last_angle = self.cm.car_angle[self.ai_indices].copy()

    def step(self, dt):
        idx = self.ai_indices
        car_pos = self.cm.car_pos[idx]
        car_vel = self.cm.car_velocity[idx]
        car_angle = self.cm.car_angle[idx]

        rays_start, rays_end = self.cm.compute_rays()
        hit_mask, hit_dist, hit_point = self.cm.raycast(rays_start, rays_end)
        ai_distances = np.clip(hit_dist[idx] / self.cm.ray_length, 0, 1)
        ai_distances = np.float_power(ai_distances, 0.1)

        speed = np.linalg.norm(car_vel, axis=1, keepdims=True) / self.cm.max_speed
        angular_vel = ((car_angle - self.last_angle)/dt).reshape(-1,1) / 180.0
        self.last_angle = car_angle.copy()

        fwd_vec = np.column_stack([np.cos(np.deg2rad(car_angle)), np.sin(np.deg2rad(car_angle))])
        speed_along_fwd = np.einsum('ij,ij->i', car_vel, fwd_vec)
        lat_vel = car_vel - fwd_vec * speed_along_fwd[:, None]
        lat_speed = np.linalg.norm(lat_vel, axis=1, keepdims=True) / self.cm.max_speed

        sp = self.cm.spline_points[:, :2]
        rel = car_pos[:, None, :] - sp[None, :, :]
        closest_idx = np.argmin(np.sum(rel**2, axis=2), axis=1)
        look_ahead = 6
        target_idx = (closest_idx + look_ahead) % len(sp)
        targets = sp[target_idx]
        dir_norm = (targets - car_pos) / np.maximum(np.linalg.norm(targets - car_pos, axis=1, keepdims=True), 1e-6)
        heading_dot = np.sum(fwd_vec * dir_norm, axis=1, keepdims=True)

        X = np.hstack([ai_distances, speed, heading_dot, angular_vel, lat_speed])
        out = self.nets.forward(X)
        throttle = np.clip(out[:,0], 0, 1)
        steer_left = out[:,1]
        steer_right = out[:,2]
        brake = np.clip(out[:,3], 0, 1)
        steer = steer_right - steer_left
        max_steer = 1.0
        steer = np.clip(steer, -max_steer, max_steer)
        self.cm.car_angle[idx] = (self.cm.car_angle[idx] + steer * self.cm.turn_speed * dt) % 360.0

        alignment = np.clip(np.einsum('ij,ij->i', fwd_vec, dir_norm), 0, 1)
        accel = fwd_vec * (throttle * alignment * self.cm.acceleration * dt)[:, None]
        accel += fwd_vec * (-brake * alignment * self.cm.brake_strength * dt)[:, None]
        self.cm.car_velocity[idx] += accel

    def evaluate(self, steps=500, dt=1/60):
        self.fitness.fill(0.0)
        self.last_target_idx.fill(0)
        idx = self.ai_indices
        sp = self.cm.spline_points[:, :2]

        for _ in range(steps):
            self.step(dt)
            self.cm.update(dt)
            car_pos = self.cm.car_pos[idx]
            rel = car_pos[:, None, :] - sp[None, :, :]
            closest_idx = np.argmin(np.sum(rel**2, axis=2), axis=1)
            progress = (closest_idx - self.last_target_idx) % len(sp)
            self.fitness += progress
            self.last_target_idx = closest_idx

    def update_fitness(self, car_indices, dt):
        ai_idx_local = np.array([np.where(self.ai_indices == ci)[0][0] for ci in car_indices])
        cm = self.cm
        sp = cm.spline_points[:, :2]
        car_pos = cm.car_pos[car_indices]
        car_vel = cm.car_velocity[car_indices]
        car_angle = cm.car_angle[car_indices]
        N = len(car_indices)

        sp = self.cm.spline_points[:, :2]
        seg_len = np.linalg.norm(np.diff(sp, axis=0), axis=1)
        self.cm.spline_cumlen = np.concatenate([[0], np.cumsum(seg_len)])
        rel = car_pos[:, None, :] - sp[None, :, :]
        dist2 = np.sum(rel**2, axis=2)
        closest_idx = np.argmin(dist2, axis=1)
        completion = self.cm.spline_cumlen[closest_idx] / self.cm.spline_cumlen[-1]

        fwd_vec = np.column_stack([np.cos(np.deg2rad(car_angle)), np.sin(np.deg2rad(car_angle))])
        rel = car_pos[:, None, :] - sp[None, :, :]
        dist2 = np.sum(rel**2, axis=2)
        closest_idx = np.argmin(dist2, axis=1)
        closest_point = sp[closest_idx]

        if not hasattr(self, 'last_target_idx'):
            self.last_target_idx = np.zeros(self.N_ai, dtype=int)
        last_idx = self.last_target_idx[ai_idx_local]
        last_point = sp[last_idx]

        move_vec = closest_point - last_point
        spline_progress = np.einsum('ij,ij->i', move_vec, fwd_vec)
        spline_progress = np.clip(spline_progress, 0, None)
        distance_to_spline = np.linalg.norm(car_pos - closest_point, axis=1)
        spline_progress *= np.exp(-distance_to_spline * 2.0)

        speed_along_fwd = np.einsum('ij,ij->i', car_vel, fwd_vec)
        if not hasattr(self, 'last_speed'):
            self.last_speed = np.zeros(self.N_ai)
        acceleration = (speed_along_fwd - self.last_speed[ai_idx_local]) / dt
        velocity_reward = np.clip(speed_along_fwd * dt, 0, None) + 0.1 * np.clip(acceleration, 0, None)
        self.last_speed[ai_idx_local] = speed_along_fwd

        lateral_vec = car_vel - fwd_vec * speed_along_fwd[:, None]
        lateral_speed = np.linalg.norm(lateral_vec, axis=1)
        lateral_penalty = lateral_speed * 0.1 * np.clip(speed_along_fwd / 5.0, 0, 1)

        collisions = np.zeros(N)
        if hasattr(cm, "collision_mask"):
            collisions = cm.collision_mask[car_indices].astype(float)
        collision_penalty = collisions * (5.0 + 0.5 * speed_along_fwd)

        self.fitness[ai_idx_local] += 0.5 * spline_progress + 0.3 * velocity_reward
        self.fitness[ai_idx_local] -= lateral_penalty + collision_penalty
        self.fitness[ai_idx_local] = completion * 100.0

        self.last_target_idx[ai_idx_local] = closest_idx.copy()

    def evolve(self, retain=0.4, mutate_rate=0.2):
        idx_sorted = np.argsort(self.fitness)[::-1]
        retain_len = max(1, int(self.N_ai * retain))
        survivors_idx = idx_sorted[:retain_len]

        new_W1 = np.zeros_like(self.nets.W1)
        new_b1 = np.zeros_like(self.nets.b1)
        new_W2 = np.zeros_like(self.nets.W2)
        new_b2 = np.zeros_like(self.nets.b2)
        new_W3 = np.zeros_like(self.nets.W3)
        new_b3 = np.zeros_like(self.nets.b3)

        new_W1[:retain_len] = self.nets.W1[survivors_idx]
        new_b1[:retain_len] = self.nets.b1[survivors_idx]
        new_W2[:retain_len] = self.nets.W2[survivors_idx]
        new_b2[:retain_len] = self.nets.b2[survivors_idx]
        new_W3[:retain_len] = self.nets.W3[survivors_idx]
        new_b3[:retain_len] = self.nets.b3[survivors_idx]

        n_children = self.N_ai - retain_len
        parents_idx = np.random.choice(survivors_idx, size=n_children)
        new_W1[retain_len:] = self.nets.W1[parents_idx]
        new_b1[retain_len:] = self.nets.b1[parents_idx]
        new_W2[retain_len:] = self.nets.W2[parents_idx]
        new_b2[retain_len:] = self.nets.b2[parents_idx]
        new_W3[retain_len:] = self.nets.W3[parents_idx]
        new_b3[retain_len:] = self.nets.b3[parents_idx]

        for param in [new_W1, new_b1, new_W2, new_b2, new_W3, new_b3]:
            mutation_mask = np.random.rand(*param.shape) < mutate_rate
            param += mutation_mask * np.random.randn(*param.shape) * 0.3

        self.nets.W1 = new_W1
        self.nets.b1 = new_b1
        self.nets.W2 = new_W2
        self.nets.b2 = new_b2
        self.nets.W3 = new_W3
        self.nets.b3 = new_b3

def draw_best_network(screen, ai_manager, position=(50,50), spacing_x=60, spacing_y=40):
    best_idx = np.argmax(ai_manager.fitness)
    best_nn = SimpleNN.__new__(SimpleNN)
    best_nn.N_net = 1
    best_nn.W1 = ai_manager.nets.W1[best_idx:best_idx+1].copy()
    best_nn.b1 = ai_manager.nets.b1[best_idx:best_idx+1].copy()
    best_nn.W2 = ai_manager.nets.W2[best_idx:best_idx+1].copy()
    best_nn.b2 = ai_manager.nets.b2[best_idx:best_idx+1].copy()
    best_nn.W3 = ai_manager.nets.W3[best_idx:best_idx+1].copy()
    best_nn.b3 = ai_manager.nets.b3[best_idx:best_idx+1].copy()

    best_idx = np.argmax(ai_manager.fitness)
    ai_idx = ai_manager.ai_indices[best_idx]

    rays_start, rays_end = ai_manager.cm.compute_rays()
    hit_mask, hit_dist, hit_point = ai_manager.cm.raycast(rays_start, rays_end)
    ai_distances = np.clip(hit_dist[ai_idx] / ai_manager.cm.ray_length, 0, 1)

    car_vel = ai_manager.cm.car_velocity[ai_idx]
    car_angle = ai_manager.cm.car_angle[ai_idx]
    fwd_vec = np.array([np.cos(np.deg2rad(car_angle)), np.sin(np.deg2rad(car_angle))])

    speed = np.linalg.norm(car_vel) / ai_manager.cm.max_speed
    speed_along_fwd = np.dot(car_vel, fwd_vec) / ai_manager.cm.max_speed
    lat_vel = car_vel - fwd_vec * np.dot(car_vel, fwd_vec)
    lat_speed = np.linalg.norm(lat_vel) / ai_manager.cm.max_speed

    angular_vel = 0
    if hasattr(ai_manager, "last_angle"):
        angular_vel = ((car_angle - ai_manager.last_angle[best_idx]) % 360) / 180.0
        ai_manager.last_angle[best_idx] = car_angle

    sp = ai_manager.cm.spline_points[:, :2]
    rel = sp - ai_manager.cm.car_pos[ai_idx]
    dist2 = np.sum(rel**2, axis=1)
    closest_idx = np.argmin(dist2)
    look_ahead = 6
    target_idx = (closest_idx + look_ahead) % len(sp)
    target_vec = sp[target_idx] - ai_manager.cm.car_pos[ai_idx]
    target_dir = target_vec / (np.linalg.norm(target_vec) + 1e-6)
    heading_dot = np.dot(fwd_vec, target_dir)

    best_nn.X = np.hstack([ai_distances, speed, heading_dot, angular_vel, lat_speed])
    draw_network(screen, best_nn, position=position, spacing_x=spacing_x, spacing_y=spacing_y)

def draw_network(screen, nn, X_input=None, position=(50,50), spacing_x=60, spacing_y=40):
    x0, y0 = position
    n_input = nn.W1.shape[2]
    n_hidden1 = nn.W1.shape[1]
    n_hidden2 = nn.W2.shape[1]
    n_output = nn.W3.shape[1]

    X_input = nn.X
    if X_input is not None:
        X_input = np.asarray(X_input).reshape(1,-1)
        h1 = np.tanh(np.einsum('nij,nj->ni', nn.W1, X_input) + nn.b1)
        h2 = np.tanh(np.einsum('nij,nj->ni', nn.W2, h1) + nn.b2)
        out = np.tanh(np.einsum('nij,nj->ni', nn.W3, h2) + nn.b3)
        activations = [X_input.flatten(), h1[0], h2[0], out[0]]
    else:
        activations = [np.zeros(n_input), np.zeros(n_hidden1), np.zeros(n_hidden2), np.zeros(n_output)]

    def layer_ys(n_nodes):
        if n_nodes == 1:
            return [0]
        return np.linspace(0, (n_nodes-1)*spacing_y, n_nodes)

    input_ys = layer_ys(n_input)
    hidden1_ys = layer_ys(n_hidden1)
    hidden2_ys = layer_ys(n_hidden2)
    output_ys = layer_ys(n_output)

    input_pos = [(x0, y0 + y) for y in input_ys]
    hidden1_pos = [(x0 + spacing_x, y0 + y) for y in hidden1_ys]
    hidden2_pos = [(x0 + 2*spacing_x, y0 + y) for y in hidden2_ys]
    output_pos = [(x0 + 3*spacing_x, y0 + y) for y in output_ys]

    for i, (x1, y1) in enumerate(input_pos):
        for j, (x2, y2) in enumerate(hidden1_pos):
            w = nn.W1[0,j,i]
            color = (0,255,0) if w>0 else (255,0,0)
            thickness = int(max(1, min(5, abs(w)*5)))
            pygame.draw.line(screen, color, (x1,y1), (x2,y2), thickness)

    for i, (x1, y1) in enumerate(hidden1_pos):
        for j, (x2, y2) in enumerate(hidden2_pos):
            w = nn.W2[0,j,i]
            color = (0,255,0) if w>0 else (255,0,0)
            thickness = int(max(1, min(5, abs(w)*5)))
            pygame.draw.line(screen, color, (x1,y1), (x2,y2), thickness)

    for i, (x1, y1) in enumerate(hidden2_pos):
        for j, (x2, y2) in enumerate(output_pos):
            w = nn.W3[0,j,i]
            color = (0,255,0) if w>0 else (255,0,0)
            thickness = int(max(1, min(5, abs(w)*5)))
            pygame.draw.line(screen, color, (x1,y1), (x2,y2), thickness)

    for layer_pos, layer_act in zip([input_pos, hidden1_pos, hidden2_pos, output_pos], activations):
        for pos, act in zip(layer_pos, layer_act):
            act_norm = (act + 1)/2
            act_norm = np.clip(act_norm, 0.0, 1.0)
            color = (int(act_norm*255), 0, int((1-act_norm)*255))
            pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), 8)
            pygame.draw.circle(screen, (0,0,0), (int(pos[0]), int(pos[1])), 8, 1)
