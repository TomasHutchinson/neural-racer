import pygame
import numpy as np
import track
from car import CarManager
from neat import AIManager
import neat

pygame.init()

screen_width, screen_height = 1000, 700
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

try:
    control_points = np.load("control_points.npy")
except FileNotFoundError:
    control_points = np.array([
        [200, 400, -90, 100],
        [400, 450, -45, 120],
        [600, 400, 0, 140],
        [650, 300, 45, 110],
        [600, 200, 90, 100],
        [400, 150, 135, 150],
        [200, 200, -180, 130],
        [150, 300, -135, 100]
    ], dtype=float)

PTS_PER_SEG = 30
spline_points = track.generate_oriented_spline(control_points, PTS_PER_SEG)
left_edge, right_edge = track.get_track_edges(spline_points, control_points, PTS_PER_SEG)

N_AI = 10
cars = CarManager(
    spline_points=spline_points,
    left_edge=left_edge,
    right_edge=right_edge,
    N_players=1,
    N_ai=N_AI
)
ai_manager = AIManager(cars, N_ai=N_AI)

generation = 0
generation_steps = 600
step_counter = 0
training_mode = False
target_dt = 1/60

running = True
while running:
    if training_mode:
        dt = target_dt
    else:
        dt = clock.tick(60) / 1000.0

    step_counter += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_t:
                training_mode = not training_mode
                print("Training mode:", training_mode)

    keys = pygame.key.get_pressed()
    cars.apply_player_controls(
        {
            "up": keys[pygame.K_UP],
            "down": keys[pygame.K_DOWN],
            "left": keys[pygame.K_LEFT],
            "right": keys[pygame.K_RIGHT]
        },
        dt
    )

    ai_manager.step(target_dt)
    ai_manager.update_fitness(ai_manager.ai_indices, dt)
    cars.update(target_dt)

    screen.fill((255,255,255))
    track_poly = np.vstack([left_edge, right_edge[::-1]])
    pygame.draw.polygon(screen, (200,200,200), track_poly)
    sp_pts = spline_points[:, :2].astype(int)
    if len(sp_pts) > 1:
        pygame.draw.lines(screen, (150,180,255), True, sp_pts.tolist(), 1)

    best_ai_local_idx = np.argmax(ai_manager.fitness)
    best_ai_global_idx = ai_manager.ai_indices[best_ai_local_idx]

    for i in range(cars.N):
        surf = pygame.Surface((cars.car_width, cars.car_height), pygame.SRCALPHA)
        if i == 0:
            color = (255,60,60)
        elif i == best_ai_global_idx:
            color = (255,215,0)
        else:
            color = (60,160,60)
        pygame.draw.rect(surf, color, (0,0,cars.car_width,cars.car_height))
        rotated = pygame.transform.rotate(surf, -cars.car_angle[i])
        rect = rotated.get_rect(center=cars.car_pos[i])
        screen.blit(rotated, rect.topleft)

    rays_start, rays_end = cars.compute_rays()
    hit_mask, hit_dist, hit_point = cars.raycast(rays_start, rays_end)
    for i in range(cars.N_rays):
        start = rays_start[0,i].astype(int)
        if hit_mask[0,i]:
            end = hit_point[0,i].astype(int)
            pygame.draw.line(screen, (255,0,0), start, end, 2)
            pygame.draw.circle(screen, (0,0,0), end, 3)
        else:
            end = rays_end[0,i].astype(int)
            pygame.draw.line(screen, (0,0,255), start, end, 2)
    
    draw_ai_rays = False
    if draw_ai_rays:
        for ai_idx in ai_manager.ai_indices:
            rays_start, rays_end = cars.compute_rays()
            hit_mask, hit_dist, hit_point = cars.raycast(rays_start, rays_end)
            for i in range(cars.N_rays):
                start = rays_start[ai_idx, i].astype(int)
                if hit_mask[ai_idx, i]:
                    end = hit_point[ai_idx, i].astype(int)
                    pygame.draw.line(screen, (0,255,0), start, end, 2)  #green for AI rays hitting
                    pygame.draw.circle(screen, (0,0,0), end, 3)
                else:
                    end = rays_end[ai_idx, i].astype(int)
                    pygame.draw.line(screen, (0,200,255), start, end, 1)  #light blue for AI rays

    font = pygame.font.Font(None, 24)
    vel0 = np.linalg.norm(cars.car_velocity[0])
    txt = font.render(f"Speed: {vel0:.2f}  Gen: {generation}  Training: {training_mode}", True, (0,0,0))
    screen.blit(txt, (10,10))

    best_idx = np.argmax(ai_manager.fitness)
    neat.draw_best_network(screen, ai_manager, position=(50,50), spacing_x=200, spacing_y=40)

    pygame.display.flip()
    pygame.display.set_caption(str(clock.get_fps()))

    if step_counter >= generation_steps:
        step_counter = 0
        print(f"Generation {generation}, best fitness: {np.max(ai_manager.fitness):.2f}")
        ai_manager.evolve()
        ai_manager.cm.reset_self()
        ai_manager.fitness.fill(0.0)
        generation += 1
