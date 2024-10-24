import functools
import math
import random
import time
from collections import defaultdict
import logging

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
from pettingzoo import ParallelEnv
import ray
import torch

# Initialisiere das Logging am Anfang des Skripts
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Stelle sicher, dass CUDA verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Klasse zur Definition einer Wand (für zukünftige Erweiterungen)
class Wall:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        width = abs(start[0] - end[0])
        height = abs(start[1] - end[1])
        self.top_left = (min(start[0], end[0]), max(start[1], end[1]))
        self.bottom_right = (
            self.top_left[0] + width,
            self.top_left[1] - height,
        )


# Klasse zur Definition eines Pinsels
class Brush:
    def __init__(self, m, w, r, l, alpha, E, I):
        self.m = m  # Masse des Pinsels
        self.w = w  # Frequenz
        self.r = r  # Exzentrizität
        self.l = l  # Länge
        self.alpha = alpha  # Winkel des Pinsels
        self.E = E  # Elastizitätsmodul
        self.I = I  # Flächenträgheitsmoment
        self.v = 0  # Geschwindigkeit
        self.theta = self.m * self.w**2 * self.r * self.l**2 * math.cos(self.alpha) / (3 * self.E * self.I)


# Klasse zur Definition eines BrushBots mit Rotationsdynamik
class BrushBot:
    def __init__(
        self,
        agent_name,
        w1,
        w2,
        x=0.0,
        y=0.0,
        mass=1.0,
        width=0.06,  # Breite der Ellipse
        height=0.03,  # Höhe der Ellipse
        seed=None,
        direction=None,
        initial_rotation=0.0,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.v_net = 0.0
        self.agent_name = agent_name  # Name des Agenten für Logging-Zwecke
        self.position = torch.tensor([x, y], dtype=torch.float32, device=device)
        # Startrichtung hinzufügen
        if direction is not None:
            self.direction = direction
        else:
            self.direction = random.uniform(0, 2 * math.pi)  # Zufällige Startrichtung
        self.dt = 0.0001  # Kleinerer Zeitschritt für langsamere Bewegung
        self.path = [
            self.position.clone().cpu().numpy()
        ]  # Klone den Tensor ohne den ursprünglichen zu beeinflussen und verwende numpy array
        self.mass = mass  # Masse des BrushBots
        self.width = width  # Breite der Ellipse
        self.height = height  # Höhe der Ellipse
        # Berechne das Trägheitsmoment der Ellipse um den Mittelpunkt
        self.inertia = (self.mass / 4) * (self.width**2 + self.height**2) / 4  # Trägheitsmoment einer Ellipse
        self.velocity = torch.zeros(2, dtype=torch.float32, device=device)  # Anfangsgeschwindigkeit Null
        self.angular_velocity = 0.0  # Anfangs-Winkelgeschwindigkeit
        self.rotation = initial_rotation  # Anfangsrotation

        # Physikalische Eigenschaften des Pinsels
        self.density_brush = 1090
        self.m = 0.01  # Masse des Motors in kg
        self.r = 0.25  # Exzentrizität
        self.l = 0.01  # Länge des Pinsels in m
        self.d = 2.4 / 1000  # Durchmesser des Pinsels in m
        self.alpha = 1.3  # Anfangswinkel des Pinsels in Radiant
        self.E = 2.1 * 10**11  # Elastizitätsmodul in Pa (aus dem Papier)
        self.I = math.pi / 4 * (self.d / 2) ** 4  # Flächenträgheitsmoment in m^4 (aus dem Papier)
        self.axis = 0.02  # Abstand zwischen den Pinseln

        # Berechne die individuellen Eigenschaften der Pinsel
        self.brush1 = Brush(self.m, w1, self.r, self.l, self.alpha, self.E, self.I)
        self.brush2 = Brush(self.m, w2, self.r, self.l, self.alpha, self.E, self.I)

        # Externer Impuls aus Kollisionen
        self.external_force = torch.zeros(2, dtype=torch.float32, device=device)
        self.external_torque = 0.0

    # Methode zur Aktualisierung der Geschwindigkeit basierend auf den Pinseln
    def update_velocity(self):
        # Aktualisiere theta basierend auf der aktuellen Frequenz
        self.brush1.theta = (
            self.m * self.brush1.w**2 * self.r * self.l**2 * math.cos(self.alpha) / (3 * self.E * self.I)
        )
        self.brush2.theta = (
            self.m * self.brush2.w**2 * self.r * self.l**2 * math.cos(self.alpha) / (3 * self.E * self.I)
        )

        # Berechnung der Pinselgeschwindigkeiten
        self.brush1.v = (self.l * math.cos(self.alpha - self.brush1.theta) - self.l * math.cos(self.alpha)) * (
            self.brush1.w / (2 * math.pi)
        )  # Linker Pinsel

        self.brush2.v = (self.l * math.cos(self.alpha - self.brush2.theta) - self.l * math.cos(self.alpha)) * (
            self.brush2.w / (2 * math.pi)
        )  # Rechter Pinsel

        # Berechnung der Netto-Vorwärtsgeschwindigkeit und der zusätzlichen Drehung
        self.v_net = (self.brush1.v + self.brush2.v) / 2
        self.angular_velocity_input = (self.brush2.v - self.brush1.v) / self.axis  # Winkelgeschwindigkeit

    # Methode zur Aktualisierung des BrushBots
    def update(self):
        # Aktualisiere die Geschwindigkeit basierend auf den Pinseln (interne Kräfte)
        # Berechne die Beschleunigung durch die internen Kräfte (Pinselmotoren)
        force_magnitude = self.v_net * self.mass / self.dt  # Vereinfachtes Modell

        # Richtung der internen Kraft
        force_direction = torch.tensor(
            [math.cos(self.direction), math.sin(self.direction)],
            dtype=torch.float32,
            device=device,
        )

        # Interne Kraft
        internal_force = force_magnitude * force_direction

        # Gesamtkräfte
        total_force = internal_force + self.external_force

        # Beschleunigung
        acceleration = total_force / self.mass

        # Aktualisiere die Geschwindigkeit
        self.velocity += acceleration * self.dt

        # Setze die externe Kraft zurück
        self.external_force = torch.zeros(2, dtype=torch.float32, device=device)

        # Aktualisiere die Position
        self.update_position()

        # Aktualisiere die Rotationsdynamik
        self.angular_velocity += (self.external_torque / self.inertia) * self.dt
        self.angular_velocity += self.angular_velocity_input  # Pinsel-induzierte Drehung
        self.direction += self.angular_velocity * self.dt
        self.direction = self.direction % (2 * math.pi)
        self.rotation += self.angular_velocity * self.dt
        self.external_torque = 0.0

        # Debugging-Ausgabe der Geschwindigkeit
        logging.info(
            f"{self.agent_name}: v_net={self.v_net}, velocity={self.velocity.cpu().numpy()}, "
            f"position={self.position.cpu().numpy()}"
        )

    # Methode zur Aktualisierung der Position basierend auf der Geschwindigkeit
    def update_position(self):
        self.position += self.velocity * self.dt
        self.path.append(self.position.clone().cpu().numpy())

    # Methode zur Erkennung von Wandkollisionen und Anpassung der Position und Geschwindigkeit
    def detect_wall_collision(self, container_size=2.0):
        collided = False
        half_size = container_size / 2

        # Speichere die Geschwindigkeit und Winkelgeschwindigkeit vor der Kollision
        vel_before = self.velocity.clone().cpu().numpy()
        omega_before = self.angular_velocity
        mass = self.mass
        inertia = self.inertia
        ke_before = 0.5 * mass * np.linalg.norm(vel_before) ** 2 + 0.5 * inertia * omega_before**2

        offset = 1e-5  # Kleiner Wert, um den Roboter leicht von der Wand zu entfernen

        # Links oder Rechts
        if self.position[0] - self.width / 2 <= -half_size:
            self.velocity[0] *= -1
            self.position[0] = -half_size + self.width / 2 + offset
            collided = True
            self.angular_velocity *= -1
        elif self.position[0] + self.width / 2 >= half_size:
            self.velocity[0] *= -1
            self.position[0] = half_size - self.width / 2 - offset
            collided = True
            self.angular_velocity *= -1

        # Oben oder Unten
        if self.position[1] - self.height / 2 <= -half_size:
            self.velocity[1] *= -1
            self.position[1] = -half_size + self.height / 2 + offset
            collided = True
            self.angular_velocity *= -1
        elif self.position[1] + self.height / 2 >= half_size:
            self.velocity[1] *= -1
            self.position[1] = half_size - self.height / 2 - offset
            collided = True
            self.angular_velocity *= -1

        if collided:
            # Geschwindigkeit und Winkelgeschwindigkeit nach der Kollision
            vel_after = self.velocity.clone().cpu().numpy()
            omega_after = self.angular_velocity
            ke_after = 0.5 * mass * np.linalg.norm(vel_after) ** 2 + 0.5 * inertia * omega_after**2

            # Impulsänderung
            impulse = mass * (vel_after - vel_before)
            # Drehimpulsänderung
            torque = inertia * (omega_after - omega_before)

            # Ausgabe der Kollisionsinformationen
            logging.info(
                f"Kollision mit der Wand erkannt für {self.agent_name} an Position {self.position.cpu().numpy()}"
            )
            logging.info(
                f"Vor der Kollision: velocity={vel_before}, angular_velocity={omega_before}, "
                f"kinetic_energy={ke_before}"
            )
            logging.info(
                f"Nach der Kollision: velocity={vel_after}, angular_velocity={omega_after}, "
                f"kinetic_energy={ke_after}"
            )
            logging.info(f"Impulsänderung: {impulse}, Drehimpulsänderung: {torque}")

        return collided

    # Methode zur Rückgabe des Bewegungspfads des BrushBots
    def get_path(self):
        return self.path

    # Methode zur Rückgabe des aktuellen Zustands
    def get_state(self):
        return [
            self.position.cpu().numpy()[0],
            self.position.cpu().numpy()[1],
            self.direction,
            self.brush1.w,
            self.brush2.w,
        ]

    # Methode zur Rückgabe der aktuellen Geschwindigkeit
    def get_velocity(self):
        return self.velocity.clone().cpu().numpy()

    # Methode zur Rückgabe der Winkelgeschwindigkeit
    def get_angular_velocity(self):
        return self.angular_velocity

    # Methode zur Rückgabe der Rotation (für Visualisierung)
    def get_rotation(self):
        return self.rotation

    # Methode zur Rückgabe der Pinsel-Frequenzen
    def get_brush_frequencies(self):
        return (self.brush1.w, self.brush2.w)

    # Methode zur Rückgabe der Masse
    def get_mass(self):
        return self.mass

    # Methode zur Rückgabe des Trägheitsmoments
    def get_inertia(self):
        return self.inertia

    # Methode zur Anwendung einer externen Kraft
    def apply_force(self, force):
        self.external_force += torch.tensor(force, dtype=torch.float32, device=device)

    # Methode zur Anwendung eines externen Drehmoments
    def apply_torque(self, torque):
        self.external_torque += torque

    # Methode zum Setzen der Position (für Überlappungskorrektur)
    def set_position(self, new_position):
        self.position = torch.tensor(new_position, dtype=torch.float32, device=device)

    # Methode zum Abrufen der Position
    def get_position(self):
        return self.position.clone().cpu().numpy()

    # Methode zum Abrufen der Ellipsenparameter
    def get_ellipse_params(self):
        return self.width, self.height, self.rotation

    # Methode zum Aktualisieren des BrushBots mit einer Aktion
    def perform_action(self, action):
        # Setze die Aktion (Frequenzen der Pinsel)
        self.brush1.w = action[0] * 100  # Skalierungsfaktor
        self.brush2.w = action[1] * 100

        # Aktualisiere die Geschwindigkeit basierend auf den neuen Frequenzen
        self.update_velocity()
        # Aktualisiere den BrushBot
        self.update()
        # Rückgabe des aktuellen Zustands
        return self.get_state()

    # Methode zum zufälligen Auswählen einer Aktion

    def sample_action(self):
        freq1 = random.randint(1, 5)
        freq2 = random.randint(1, 5)
        while freq2 == freq1:
            freq2 = random.randint(1, 5)
        return [freq1, freq2]

    # def sample_action(self):
    #    if self.agent_name == "robot_1":
    #        freq1 = 3
    #        freq2 = 3
    #   else:
    #       freq1 = 1
    #        freq2 = 1
    #    return [freq1, freq2]

    def set_velocity(self, new_velocity):
        self.velocity = torch.tensor(new_velocity, dtype=torch.float32, device=device)

    def set_angular_velocity(self, new_angular_velocity):
        self.angular_velocity = new_angular_velocity


# Ray Actor Klasse
@ray.remote
class RayBrushBot:
    def __init__(
        self,
        agent_name,
        w1,
        w2,
        x=0.0,
        y=0.0,
        mass=1.0,
        width=0.06,
        height=0.03,
        seed=None,
        direction=None,
        initial_rotation=0.0,
    ):
        self.bot = BrushBot(
            agent_name,
            w1,
            w2,
            x,
            y,
            mass,
            width,
            height,
            seed,
            direction,
            initial_rotation,
        )

    def perform_action(self, action):
        return self.bot.perform_action(action)

    def get_state(self):
        return self.bot.get_state()

    def get_velocity(self):
        return self.bot.get_velocity()

    def get_inertia(self):
        return self.bot.get_inertia()

    def get_angular_velocity(self):
        return self.bot.get_angular_velocity()

    def get_rotation(self):
        return self.bot.get_rotation()

    def apply_force(self, force):
        self.bot.apply_force(force)

    def apply_torque(self, torque):
        self.bot.apply_torque(torque)

    def detect_wall_collision(self, container_size=2.0):
        return self.bot.detect_wall_collision(container_size)

    def get_path(self):
        return self.bot.get_path()

    def get_mass(self):
        return self.bot.get_mass()

    def get_brush_frequencies(self):
        return self.bot.get_brush_frequencies()

    def sample_action(self):
        return self.bot.sample_action()

    def set_position(self, new_position):
        self.bot.set_position(new_position)

    def get_position(self):
        return self.bot.get_position()

    def get_ellipse_params(self):
        return self.bot.get_ellipse_params()

    def set_dt(self, dt):
        self.bot.dt = dt

    def get_agent_name(self):
        return self.bot.agent_name

    def set_velocity(self, new_velocity):
        self.bot.set_velocity(new_velocity)

    def set_angular_velocity(self, new_angular_velocity):
        self.bot.set_angular_velocity(new_angular_velocity)


# Hilfsfunktion für das Kreuzprodukt eines Skalars und eines Vektors in 2D
def cross2d_scalar_vector(scalar, vector):
    return scalar * np.array([-vector[1], vector[0]])


# Hilfsfunktion für das Kreuzprodukt zweier Vektoren in 2D
def cross2d_vector_vector(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


# Klasse zur Definition der Umgebung für den BrushBot-Schwarm mit Ray
class BrushBotSwarmParallelEnv(ParallelEnv):
    metadata = {"name": "brushbot_environment", "render_modes": ["human"]}

    def __init__(
        self,
        num_robots=1,
        use_same_seed=False,
        container_size=2.0,
        cell_size=0.1,
        show_grid=True,
    ):
        super().__init__()
        self.num_robots = num_robots  # Anzahl der Roboter
        self.agents = [f"robot_{i}" for i in range(num_robots)]  # Liste der Roboter
        self.possible_agents = self.agents[:]  # Liste der möglichen Roboter
        self.use_same_seed = use_same_seed
        self.container_size = container_size  # Größe des quadratischen Behälters
        self.cell_size = cell_size  # Größe der Zellen für die räumliche Partitionierung
        self.show_grid = show_grid  # Flag, um das Gitter anzuzeigen

        # Seed-Initialisierung
        if self.use_same_seed:
            self.seeds = [5] * num_robots  # Verwende denselben Seed für alle Roboter
        else:
            self.seeds = [random.randint(0, 10000) for _ in range(num_robots)]  # Zufällige Seeds für alle Roboter

        # Berechne die Anzahl der Zellen in jeder Richtung
        self.num_cells = int(container_size / cell_size)

        # Initialisierung des Plots
        plt.ion()
        self.figure, self.ax = plt.subplots()
        self.colors = plt.cm.rainbow(np.linspace(0, 1, num_robots))  # Farben für jeden Roboter generieren
        self.ax.grid(True)
        half_size = self.container_size / 2
        self.ax.set_xlim(-half_size, half_size)
        self.ax.set_ylim(-half_size, half_size)
        self.ax.set_xlabel("X-Position")
        self.ax.set_ylabel("Y-Position")
        self.ax.set_title("BrushBot Schwarm Bewegungspfad")

        # Zeichne das Gitter, falls aktiviert
        if self.show_grid:
            self.draw_grid()

        # Initialisierung der Ellipsen
        self.ellipses = []
        for color in self.colors:
            ellipse = Ellipse(
                xy=(0, 0),
                width=0.06,  # Neue Breite der Ellipse
                height=0.03,  # Neue Höhe der Ellipse
                angle=0,  # Anfangswinkel
                edgecolor=color,
                facecolor="none",
                linewidth=2,
            )
            self.ax.add_patch(ellipse)
            self.ellipses.append(ellipse)

        # Initialisierung der BrushBots als Ray Actors
        self.robots = []
        self.wall = Wall(
            (-half_size, -half_size), (half_size, half_size)
        )  # Nicht direkt verwendet, aber für zukünftige Erweiterungen

        # Definition der Aktions- und Beobachtungsräume
        self.action_spaces = {agent: gym.spaces.MultiDiscrete([3, 3]) for agent in self.agents}
        low = np.array([-np.inf, -np.inf, 0.0, 0, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, 2 * math.pi, 300, 300], dtype=np.float32)
        self.observation_spaces = {agent: gym.spaces.Box(low=low, high=high, dtype=np.float32) for agent in self.agents}

        # Liste für die Geschwindigkeits-Pfeile
        self.arrows = []

        # Initialisierung der Zeitlisten und Kollisionszählung
        self.initialize_time_lists()

    def draw_grid(self):
        # Zeichne vertikale und horizontale Gitterlinien
        for i in range(1, self.num_cells):
            # Vertikale Linien
            x = -self.container_size / 2 + i * self.cell_size
            self.ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            # Horizontale Linien
            y = -self.container_size / 2 + i * self.cell_size
            self.ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    def toggle_grid(self, show=True):
        self.show_grid = show
        if show:
            self.draw_grid()
        else:
            # Entferne alle bestehenden Gitterlinien
            for line in self.ax.get_lines():
                if line.get_linestyle() == "--" and line.get_color() == "gray":
                    line.remove()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # Methode zur Initialisierung der Roboter als Ray Actors
    def init_robots(self, start_positions=None, start_orientations=None, start_rotations=None):
        robots = []
        for i in range(self.num_robots):
            half_size = self.container_size / 2
            # Startposition festlegen
            if start_positions and i < len(start_positions) and start_positions[i] is not None:
                x, y = start_positions[i]
            else:
                # Initialisiere BrushBots innerhalb des Behälters mit einem Puffer
                x = random.uniform(-half_size + 0.05, half_size - 0.05)
                y = random.uniform(-half_size + 0.05, half_size - 0.05)

            # Setze die Richtung
            if start_orientations and i < len(start_orientations) and start_orientations[i] is not None:
                direction = start_orientations[i]
            else:
                direction = None  # Zufällige Richtungen

            # Anfangsrotation setzen
            if start_rotations and i < len(start_rotations) and start_rotations[i] is not None:
                initial_rotation = start_rotations[i]
            else:
                initial_rotation = 0.0  # Standardrotation

            w1 = 100
            w2 = 100
            mass = 1.0
            width = 0.06  # Breite der Ellipse
            height = 0.03  # Höhe der Ellipse
            seed = self.seeds[i] if self.use_same_seed else self.seeds[i]
            agent_name = self.agents[i]
            robot = RayBrushBot.remote(
                agent_name,
                w1,
                w2,
                x,
                y,
                mass,
                width,
                height,
                seed,
                direction,
                initial_rotation,
            )
            robots.append(robot)
        return robots

    # Reset-Methode zur Initialisierung der Umgebung
    def reset(
        self,
        seed=None,
        options=None,
        center=None,
        start_positions=None,
        start_orientations=None,
        start_rotations=None,
    ):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        if self.use_same_seed:
            # Verwende denselben Seed für alle Roboter
            self.seeds = [seed if seed is not None else random.randint(0, 10000)] * self.num_robots
        else:
            # Zufällige Seeds für alle Roboter
            self.seeds = [random.randint(0, 10000) for _ in range(self.num_robots)]

        # Initialisierung der Roboter mit Startpositionen, Orientierungen und Rotationen
        self.robots = self.init_robots(
            start_positions=start_positions,
            start_orientations=start_orientations,
            start_rotations=start_rotations,
        )

        # Alle Agenten sind zu Beginn aktiv
        self.agents = self.possible_agents[:]

        # Erstelle die Anfangsbeobachtungen für alle Roboter
        observations = {}
        futures = [robot.get_state.remote() for robot in self.robots]
        states = ray.get(futures)
        for i, agent in enumerate(self.agents):
            observations[agent] = states[i]
        infos = {agent: {} for agent in self.agents}

        # Reset der Ellipsenpositionen
        for ellipse in self.ellipses:
            ellipse.center = (0, 0)
            ellipse.angle = 0

        # Entferne vorherige Pfeile
        for arrow in self.arrows:
            arrow.remove()
        self.arrows = []

        # Reset der Zeitlisten und Kollisionen
        self.initialize_time_lists()

        return observations, infos

    # Methode zur Berechnung des adaptiven Zeitschritts
    def compute_adaptive_timestep(self):
        # Berechne den minimalen Abstand zwischen allen BrushBots
        positions = []
        futures = [robot.get_position.remote() for robot in self.robots]
        positions = ray.get(futures)
        min_distance = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                delta_pos = np.array(positions[i]) - np.array(positions[j])
                distance = np.linalg.norm(delta_pos)
                if distance < min_distance and distance > 0:
                    min_distance = distance
        # Berechne den maximalen Betrag der Geschwindigkeiten
        velocities = []
        futures = [robot.get_velocity.remote() for robot in self.robots]
        velocities = ray.get(futures)
        max_speed = max([np.linalg.norm(v) for v in velocities])
        # Setze den Zeitschritt entsprechend
        if max_speed > 0 and min_distance > 0:
            dt = min(0.01, (min_distance / (2 * max_speed)) * 0.5)
        else:
            dt = 0.01
        # Aktualisiere den Zeitschritt in allen BrushBots
        for robot in self.robots:
            robot.set_dt.remote(dt)

    # Schritt-Methode zur Aktualisierung der Umgebung - Parallele Aktualisierung mit Ray
    def step_parallel(self, actions):
        # Berechne den adaptiven Zeitschritt
        self.compute_adaptive_timestep()

        # Parallele Aktualisierung
        parallel_start_time = time.time()
        # Sende Aktionen an alle Roboter
        futures = []
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            futures.append(self.robots[i].perform_action.remote(action))
        # Empfange Zustände von allen Robotern
        states = ray.get(futures)
        observations = {}
        rewards = {}
        truncations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = states[i]
            # Belohnungssystem: Belohnung für keine Kollision
            collision_response = ray.get(self.robots[i].detect_wall_collision.remote())
            if collision_response:
                rewards[agent] = -1.0
                logging.info(f"{agent} kollidierte mit der Wand.")
                # Ändere die Farbe der Ellipse, um die Kollision visuell hervorzuheben
                self.ellipses[i].set_edgecolor("black")
            else:
                rewards[agent] = 1.0
                # Setze die Originalfarbe zurück
                self.ellipses[i].set_edgecolor(self.colors[i])
            truncations[agent] = False
            infos[agent] = {}
        parallel_end_time = time.time()
        parallel_elapsed_time = parallel_end_time - parallel_start_time
        self.parallel_times.append(parallel_elapsed_time)

        # Kollisionserkennung und -reaktion
        self.handle_collisions()

        return observations, rewards, truncations, infos

    # Methode zur Kollisionserkennung und -reaktion mit räumlicher Partitionierung
    def handle_collisions(self):
        # Gitter-basierte räumliche Partitionierung
        grid = defaultdict(list)
        half_size = self.container_size / 2
        positions = []
        ellipse_params = []
        # Erhalte die Positionen und Ellipsenparameter aller Roboter
        futures = [robot.get_state.remote() for robot in self.robots]
        states = ray.get(futures)
        futures = [robot.get_ellipse_params.remote() for robot in self.robots]
        ellipse_params = ray.get(futures)
        for i, robot in enumerate(self.robots):
            x, y = states[i][0], states[i][1]
            positions.append((x, y))
            width, height, rotation = ellipse_params[i]
            # Bestimme die Zellen, in denen der BrushBot liegt
            cell_x = int((x + half_size) / self.cell_size)
            cell_y = int((y + half_size) / self.cell_size)
            grid[(cell_x, cell_y)].append(i)

        # Erstelle eine Liste aller potenziellen Kollisionen
        collision_pairs = set()
        for (cell_x, cell_y), robot_indices in grid.items():
            # Liste der benachbarten Zellen inklusive der aktuellen Zelle
            neighboring_cells = [(cell_x + dx, cell_y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            for idx in robot_indices:
                for neighbor in neighboring_cells:
                    for jdx in grid.get(neighbor, []):
                        if jdx <= idx:
                            continue  # Vermeide doppelte Überprüfungen und Selbstkollisionen
                        collision_pairs.add((idx, jdx))

        # Verarbeite alle Kollisionen
        for idx, jdx in collision_pairs:
            # Hole die Positionen und Ellipsenparameter der BrushBots
            pos_i = np.array(positions[idx])
            pos_j = np.array(positions[jdx])
            width_i, height_i, rotation_i = ellipse_params[idx]
            width_j, height_j, rotation_j = ellipse_params[jdx]

            # Überprüfe Kollision zwischen den Ellipsen
            if self.check_ellipse_collision(pos_i, width_i, height_i, rotation_i, pos_j, width_j, height_j, rotation_j):
                # Erhöhe den Kollisionszähler
                self.collision_count += 1
                logging.info(
                    f"Kollision zwischen {self.agents[idx]} und {self.agents[jdx]} an Positionen {pos_i} und {pos_j}"
                )

                # Führe die elastische Kollision mit Rotationsdynamik durch
                self.elastic_collision(self.robots[idx], self.robots[jdx], idx, jdx, pos_i, pos_j)

    # Methode zur Überprüfung der Kollision zwischen zwei Ellipsen
    def check_ellipse_collision(self, pos1, w1, h1, rot1, pos2, w2, h2, rot2):
        # Konvertiere die Rotationen in Grad
        angle1 = np.degrees(rot1)
        angle2 = np.degrees(rot2)

        # Verschiebe die zweite Ellipse relativ zur ersten
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]

        # Erstelle Transformationsmatrizen
        R1 = self.rotation_matrix(-angle1)
        R2 = self.rotation_matrix(-angle1 - angle2)

        # Transformiere die Koordinaten
        dx1, dy1 = np.dot(R1, np.array([dx, dy]))

        # Berechne die Summe der Halbachsen
        aw = (w1 + w2) / 2
        ah = (h1 + h2) / 2

        # Prüfe auf Kollision
        if ((dx1 / aw) ** 2 + (dy1 / ah) ** 2) <= 1:
            return True
        else:
            return False

    def rotation_matrix(self, angle_degrees):
        theta = np.radians(angle_degrees)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Methode zur Berechnung der elastischen Kollision mit Rotationsdynamik
    def elastic_collision(self, bot_i, bot_j, idx_i, idx_j, pos_i, pos_j):
        agent_i = self.agents[idx_i]
        agent_j = self.agents[idx_j]

        # Hole Geschwindigkeiten und Winkelgeschwindigkeiten vor der Kollision
        vel_i_before = ray.get(bot_i.get_velocity.remote())
        vel_j_before = ray.get(bot_j.get_velocity.remote())
        omega_i_before = ray.get(bot_i.get_angular_velocity.remote())
        omega_j_before = ray.get(bot_j.get_angular_velocity.remote())
        mass_i = ray.get(bot_i.get_mass.remote())
        mass_j = ray.get(bot_j.get_mass.remote())
        inertia_i = ray.get(bot_i.get_inertia.remote())
        inertia_j = ray.get(bot_j.get_inertia.remote())

        # Berechne kinetische Energien vor der Kollision
        ke_i_before = 0.5 * mass_i * np.linalg.norm(vel_i_before) ** 2 + 0.5 * inertia_i * omega_i_before**2
        ke_j_before = 0.5 * mass_j * np.linalg.norm(vel_j_before) ** 2 + 0.5 * inertia_j * omega_j_before**2

        # Berechne Kollisionsnormalen
        delta_pos = np.array(pos_i) - np.array(pos_j)
        distance = np.linalg.norm(delta_pos)
        collision_normal = delta_pos / distance if distance != 0 else np.array([1, 0])

        # Kontaktpunkt ist der Mittelpunkt zwischen den beiden Ellipsen
        contact_point = (np.array(pos_i) + np.array(pos_j)) / 2

        # Relative Positionen zum Kontaktpunkt
        r_i = contact_point - np.array(pos_i)
        r_j = contact_point - np.array(pos_j)

        # Relative Geschwindigkeit am Kontaktpunkt
        vel_i_contact = vel_i_before + cross2d_scalar_vector(omega_i_before, r_i)
        vel_j_contact = vel_j_before + cross2d_scalar_vector(omega_j_before, r_j)
        delta_vel = vel_i_contact - vel_j_contact

        # Berechne Impuls
        vel_along_normal = np.dot(delta_vel, collision_normal)
        if vel_along_normal > 0:
            # Roboter bewegen sich voneinander weg
            return

        restitution = 1  # Elastizitätskoeffizient
        numerator = -(1 + restitution) * vel_along_normal
        denominator = (
            1 / mass_i
            + 1 / mass_j
            + (cross2d_vector_vector(r_i, collision_normal) ** 2) / inertia_i
            + (cross2d_vector_vector(r_j, collision_normal) ** 2) / inertia_j
        )
        impulse_scalar = numerator / denominator

        # Impulsvektor
        impulse = impulse_scalar * collision_normal

        # Berechne die neuen Geschwindigkeiten und Winkelgeschwindigkeiten
        vel_i_after = vel_i_before + (impulse / mass_i)
        vel_j_after = vel_j_before - (impulse / mass_j)
        omega_i_after = omega_i_before + (cross2d_vector_vector(r_i, impulse) / inertia_i)
        omega_j_after = omega_j_before + (cross2d_vector_vector(r_j, -impulse) / inertia_j)

        # Setze die neuen Geschwindigkeiten und Winkelgeschwindigkeiten
        bot_i.set_velocity.remote(vel_i_after)
        bot_j.set_velocity.remote(vel_j_after)
        bot_i.set_angular_velocity.remote(omega_i_after)
        bot_j.set_angular_velocity.remote(omega_j_after)

        # Berechne kinetische Energien nach der Kollision
        ke_i_after = 0.5 * mass_i * np.linalg.norm(vel_i_after) ** 2 + 0.5 * inertia_i * omega_i_after**2
        ke_j_after = 0.5 * mass_j * np.linalg.norm(vel_j_after) ** 2 + 0.5 * inertia_j * omega_j_after**2

        # Ausgabe der Kollisionsinformationen
        logging.info(f"Kollision zwischen {agent_i} und {agent_j}:")
        logging.info("Vor der Kollision:")
        logging.info(
            f"  {agent_i}: velocity={vel_i_before}, angular_velocity={omega_i_before}, " f"kinetic_energy={ke_i_before}"
        )
        logging.info(
            f"  {agent_j}: velocity={vel_j_before}, angular_velocity={omega_j_before}, " f"kinetic_energy={ke_j_before}"
        )
        logging.info(
            f"Angewendeter Impuls: {impulse}, Drehmomente: {cross2d_vector_vector(r_i, impulse)}, "
            f"{cross2d_vector_vector(r_j, -impulse)}"
        )
        logging.info("Nach der Kollision:")
        logging.info(
            f"  {agent_i}: velocity={vel_i_after}, angular_velocity={omega_i_after}, " f"kinetic_energy={ke_i_after}"
        )
        logging.info(
            f"  {agent_j}: velocity={vel_j_after}, angular_velocity={omega_j_after}, " f"kinetic_energy={ke_j_after}"
        )

    # Render-Methode zur Visualisierung der Umgebung
    def render(self, mode="human"):
        futures = [robot.get_state.remote() for robot in self.robots]
        states = ray.get(futures)
        rotations = ray.get([robot.get_rotation.remote() for robot in self.robots])
        for i, robot in enumerate(self.robots):
            state = states[i]
            x, y, direction, _, _ = state
            ellipse = self.ellipses[i]
            ellipse.center = (x, y)
            ellipse.angle = np.degrees(rotations[i])  # Rotation des BrushBots

            # Geschwindigkeitsvektor abrufen
            velocity = ray.get(robot.get_velocity.remote())
            dx, dy = velocity[0] * 0.1, velocity[1] * 0.1  # Skalierungsfaktor anpassen

            # Entferne vorherige Pfeile, falls vorhanden
            if len(self.arrows) > i:
                self.arrows[i].remove()

            # Füge einen neuen Pfeil hinzu
            arrow = FancyArrowPatch(
                (x, y),
                (x + dx, y + dy),
                arrowstyle="->",
                mutation_scale=10,
                color=self.colors[i],
            )
            self.ax.add_patch(arrow)
            if len(self.arrows) <= i:
                self.arrows.append(arrow)
            else:
                self.arrows[i] = arrow

        # Aktualisiere die Position der Ellipsen und Pfeile
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        # Zeichne die Wand (nur einmal hinzufügen, um Mehrfachzeichnungen zu vermeiden)
        if not hasattr(self, "wall_patch"):
            half_size = self.container_size / 2
            self.ax.set_xlim(-half_size, half_size)
            self.ax.set_ylim(-half_size, half_size)
            self.wall_patch = plt.Rectangle(
                (-half_size, -half_size),
                self.container_size,
                self.container_size,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            self.ax.add_patch(self.wall_patch)

        # Zeichne die Pfade der Roboter
        futures = [robot.get_path.remote() for robot in self.robots]
        paths = ray.get(futures)
        for i, path in enumerate(paths):
            path = np.array(path)
            self.ax.plot(path[:, 0], path[:, 1], color=self.colors[i], marker="o", markersize=2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.MultiDiscrete([5, 5])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        low = np.array([-np.inf, -np.inf, 0.0, 0, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, 2 * math.pi, 600, 600], dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def close(self):
        plt.close(self.figure)
        # Terminiere alle Actors
        for robot in self.robots:
            ray.kill(robot)

    # Listen zur Speicherung der Zeiten und Kollisionszählung
    def initialize_time_lists(self):
        self.parallel_times = []
        self.sequential_times = []
        self.collision_count = 0
        # Massen der Agenten (hier alle gleich 1.0, kann angepasst werden)
        self.agents_masses = [1.0 for _ in range(self.num_robots)]

    # Schritt-Methode - General
    def step(self, actions):
        return self.step_parallel(actions)


# Main
if __name__ == "__main__":
    # Initialisiere Ray
    ray.init()

    # Setze die Anzahl der Roboter auf 4
    env = BrushBotSwarmParallelEnv(num_robots=4, use_same_seed=False, container_size=2.0, cell_size=0.1, show_grid=True)
    env.initialize_time_lists()

    # Startpositionen der Roboter
    start_positions = [
        # (-0.5, 0.0),  # Roboter 0
        # (0.5, 0.0),  # Roboter 1
        # (0.0, -0.5),  # Roboter 2
        # (0.0, 0.5),  # Roboter 3
        None,
        None,
        None,
        None,
    ]

    # Startorientierungen der Roboter (sie schauen zur Mitte)
    start_orientations = [
        # 0.0,  # Roboter 0 blickt nach rechts
        # math.pi,  # Roboter 1 blickt nach links
        # math.pi / 2,  # Roboter 2 blickt nach oben
        # 3 * math.pi / 2,  # Roboter 3 blickt nach unten
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # Startrotationen der Roboterkörper (optional)
    start_rotations = [
        # 0.0,  # Roboter 0
        # math.pi,  # Roboter 1
        # math.pi / 2,  # Roboter 2
        # 3 * math.pi / 2,  # Roboter 3
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # Initialisierung der Umgebung und der Roboter
    observations, infos = env.reset(
        seed=37,
        start_positions=start_positions,
        start_orientations=start_orientations,
        start_rotations=start_rotations,
    )
    total_rewards = dict.fromkeys(env.agents, 0)

    # Anzahl der Iterationen
    num_iterations = 1000  # Anpassen, um die Simulation zu steuern

    step = 0
    while step < num_iterations and env.agents:
        # Aktionen für alle Agenten
        actions = {}
        for agent in env.agents:
            if agent == "robot_1":
                freq_left = random.randint(1, 5)  # Frequenz für die linke Bürste
                freq_right = random.randint(1, 5)  # Frequenz für die rechte Bürste
                actions[agent] = [freq_left, freq_right]
            else:
                actions[agent] = ray.get(env.robots[env.agents.index(agent)].sample_action.remote())

        observations, rewards, truncations, infos = env.step(actions)

        # Ausgabe der aktuellen Aktionen und Beobachtungen für jeden Roboter
        print(f"Step {step}:")
        for agent in env.agents:
            print(f"{agent} - Action: {actions[agent]}, Observation: {observations[agent]}")

        for agent, reward in rewards.items():
            total_rewards[agent] += reward
            if step % 10 == 0:
                env.render()
        step += 1

        if not env.agents:
            break

    env.render()
    env.close()

    # Shutdown Ray
    ray.shutdown()

    # Optional: Kollisionszählung ausgeben
    print(f"Total collisions: {env.collision_count}")
