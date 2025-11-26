import pygame
import numpy as np

pygame.init()
screen_width, screen_height = 1000, 700
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Spline Editor with Track Visualization")
clock = pygame.time.Clock()

# COLORS
WHITE, GRAY, RED, BLUE, GREEN = (255, 255, 255), (200, 200, 200), (255, 0, 0), (0, 0, 255), (0, 255, 0)

# Load control points or create default
try:
    control_points = np.load("control_points.npy")
except FileNotFoundError:
    control_points = np.array([
        [200, 400, -90, 100],  # (x, y, rotation, track width)
        [400, 450, -45, 120],
        [600, 400, 0, 140],
        [650, 300, 45, 110],
        [600, 200, 90, 100],
        [400, 150, 135, 150],
        [200, 200, -180, 130],
        [150, 300, -135, 100]
    ], dtype=float)

selected_idx = None
font = pygame.font.SysFont(None, 24)

# -----------------------------
# SPLINE FUNCTIONS
# -----------------------------
def catmull_rom_spline_oriented(P0, P1, P2, P3, n_points=30):
    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    a = -0.5 * P0[:2] + 1.5 * P1[:2] - 1.5 * P2[:2] + 0.5 * P3[:2]
    b = P0[:2] - 2.5 * P1[:2] + 2 * P2[:2] - 0.5 * P3[:2]
    c = -0.5 * P0[:2] + 0.5 * P2[:2]
    d = P1[:2]
    pos = a * t**3 + b * t**2 + c * t + d

    def shortest_angle(a1, a2): return a1 + ((a2 - a1 + 180) % 360 - 180)
    rot1 = shortest_angle(P0[2], P1[2])
    rot2 = shortest_angle(rot1, P2[2])
    rot3 = shortest_angle(rot2, P3[2])
    r_a = -0.5 * P0[2] + 1.5 * rot1 - 1.5 * rot2 + 0.5 * rot3
    r_b = P0[2] - 2.5 * rot1 + 2 * rot2 - 0.5 * rot3
    r_c = -0.5 * P0[2] + 0.5 * rot2
    r_d = rot1
    rot = r_a * t**3 + r_b * t**2 + r_c * t + r_d
    return np.hstack([pos, rot])

def generate_oriented_spline(control_points, n_points_per_segment=30):
    spline_points = []
    n = len(control_points)
    
    # Loop through each control point and generate spline segments
    for i in range(n):
        # Use modulo to loop around to create circular indexing for the spline
        P0, P1, P2, P3 = control_points[i-1], control_points[i], control_points[(i+1) % n], control_points[(i+2) % n]
        
        # Generate spline points for this segment
        segment_points = catmull_rom_spline_oriented(P0, P1, P2, P3, n_points_per_segment)
        spline_points.append(segment_points)
    
    # Combine all spline points into a single array
    return np.vstack(spline_points)

# -----------------------------
# Interpolate Track Widths Along the Spline Path
# -----------------------------
def interpolate_widths_along_spline(control_points, n_points_per_segment):
    n_control_points = len(control_points)
    width_per_spline_point = []

    # For each spline segment between control points, interpolate track widths
    for i in range(n_control_points):  # Change the range to go through all control points
        p1 = control_points[i]
        p2 = control_points[(i + 1) % n_control_points]  # Ensure wraparound with modulo
        
        # Interpolate track widths between control points for all points in this segment
        segment_widths = np.linspace(p1[3], p2[3], n_points_per_segment)
        width_per_spline_point.extend(segment_widths)
    
    # Ensure the number of width points matches the number of spline points
    expected_length = n_control_points * n_points_per_segment
    if len(width_per_spline_point) != expected_length:
        raise ValueError(f"Width interpolation mismatch: {len(width_per_spline_point)} widths, expected {expected_length} widths.")
    
    return np.array(width_per_spline_point)
# -----------------------------
# SMOOTHED TRACK EDGES FUNCTION
# -----------------------------
def get_track_edges(spline_points, control_points, n_points_per_segment=30):
    left_edge, right_edge = [], []

    # Interpolate track widths along the entire spline path
    width_per_spline_point = interpolate_widths_along_spline(control_points, n_points_per_segment)

    # Ensure width_per_spline_point matches the number of spline points
    if len(width_per_spline_point) != len(spline_points):
        raise ValueError(f"Mismatch between number of spline points ({len(spline_points)}) and width points ({len(width_per_spline_point)})")

    # Calculate edges for each spline point
    for i, (x, y, angle) in enumerate(spline_points):
        # Get the interpolated width for the current spline point
        width = width_per_spline_point[i]

        # Calculate the perpendicular direction based on the angle
        rad = np.radians(angle)
        perp = np.array([np.sin(rad), np.cos(rad)]) * (width / 2)
        left_edge.append([x + perp[0], y + perp[1]])
        right_edge.append([x - perp[0], y - perp[1]])

    return np.array(left_edge), np.array(right_edge)

# -----------------------------
# DRAW FUNCTIONS
# -----------------------------
def draw_control_points():
    for i, (x, y, angle, width) in enumerate(control_points):
        color = RED if i == selected_idx else BLUE
        pygame.draw.circle(screen, color, (int(x), int(y)), 8)
        end_x = x + np.cos(np.radians(angle)) * 30
        end_y = y - np.sin(np.radians(angle)) * 30
        pygame.draw.line(screen, GREEN, (x, y), (end_x, end_y), 2)

def draw_spline_and_track():
    spline_points = generate_oriented_spline(control_points, 30)
    left_edge, right_edge = get_track_edges(spline_points, control_points)
    track_poly = np.vstack([left_edge, right_edge[::-1]])
    pygame.draw.polygon(screen, GRAY, track_poly, 0)
    for i in range(len(spline_points) - 1):
        pygame.draw.line(screen, GREEN, spline_points[i, :2], spline_points[i + 1, :2], 2)

# -----------------------------
# MAIN LOOP
# -----------------------------
running = True
while running:
    dt = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            for i, (x, y, r, w) in enumerate(control_points):
                if np.hypot(mx - x, my - y) < 10:
                    selected_idx = i
                    break

        elif event.type == pygame.MOUSEBUTTONUP:
            selected_idx = None

        elif event.type == pygame.KEYDOWN:
            if selected_idx is not None:
                if event.key == pygame.K_UP: control_points[selected_idx, 1] -= 5
                if event.key == pygame.K_DOWN: control_points[selected_idx, 1] += 5
                if event.key == pygame.K_LEFT: control_points[selected_idx, 0] -= 5
                if event.key == pygame.K_RIGHT: control_points[selected_idx, 0] += 5
                if event.key == pygame.K_a: control_points[selected_idx, 2] += 5
                if event.key == pygame.K_d: control_points[selected_idx, 2] -= 5
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    control_points[selected_idx, 3] += 10
                if event.key == pygame.K_MINUS:
                    control_points[selected_idx, 3] = max(10, control_points[selected_idx, 3] - 10)

            if event.key == pygame.K_s:
                np.save("control_points.npy", control_points)
                print("Control points saved!")
            if event.key == pygame.K_l:
                try:
                    control_points = np.load("control_points.npy")
                    print("Control points loaded!")
                except FileNotFoundError:
                    print("No saved control points found.")

    # Dragging
    if selected_idx is not None and pygame.mouse.get_pressed()[0]:
        mx, my = pygame.mouse.get_pos()
        control_points[selected_idx, 0] = mx
        control_points[selected_idx, 1] = my

    # DRAW
    screen.fill(WHITE)
    draw_spline_and_track()
    draw_control_points()

    screen.blit(font.render("Drag points. Arrow keys to move. A/D rotate. S to save, L to load.", True, (0, 0, 0)), (10, 40))
    pygame.display.flip()

pygame.quit()

