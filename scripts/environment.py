import gymnasium as gym
from gymnasium import spaces
from scripts import *
import os
import numpy as np
import pygame

CELL_SIZE = 35
GRID_WIDTH, GRID_HEIGHT = 20, 20
WIDTH = GRID_WIDTH * CELL_SIZE
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE
STATUS_BAR = False
BAR_HEIGHT = 2*CELL_SIZE  if STATUS_BAR else 0 # Height of the top status bar
TOTAL_HEIGHT = GAME_HEIGHT + BAR_HEIGHT
N_OBSTACLES = 10
FPS = 10

RESOURCES_PATH = "scripts/resources/"

SEED = 0

class VacuumWorld(gym.Env):
    """
    Simple grid-world vacuum cleaner environment.

    The environment contains:
        - a vacuum cleaner
        - walls
        - dirt cells

    Actions:
        0 = UP
        1 = DOWN
        2 = LEFT
        3 = RIGHT
        4 = IDLE
        5 = SUCK

    The goal is to clean every dirty cell.

    Coordinates
    -----------
    We consistently use (x, y):

        x -> column -> left/right
        y -> row    -> up/down

    Thus:
        (0, 0) = top-left cell

    For numpy arrays, the conversion is:

        grid[y, x]
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode=RenderMode.HUMAN, observation_type=ObservationType.GRID, max_step=600, difficulty=0, n_obstacles=10, n_dirt=1, **kwargs):
    
        super().__init__()

        # Convert Enum -> string if necessary
        if isinstance(render_mode, RenderMode):
            render_mode = render_mode.value

        if isinstance(observation_type, ObservationType):
            observation_type = observation_type.value

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. "
                f"Expected one of {self.metadata['render_modes']}"
            )

        if observation_type not in {"grid", "image"}:
            raise ValueError(
                f"Invalid observation_type: {observation_type}"
            )

        self.render_mode = render_mode
        self.observation_type = observation_type

        self.max_step = max_step
        self.difficulty = max(0, min(difficulty, 3))

        self.n_obstacles = n_obstacles
        self.n_dirt = n_dirt

        self.status_bar = STATUS_BAR

        self.action_space = spaces.Discrete(6) # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDLE, 5=SUCK

        if self.observation_type == ObservationType.IMAGE :
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(TOTAL_HEIGHT, WIDTH, 3), dtype=np.uint8
            )
        else:
            # 0 = clean floor
            # 1 = dirt
            # 2 = wall
            # 3 = vacuum
            self.observation_space = spaces.Box(
                low=0, high=4, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8
            )

        self.window = None
        self.canvas = None
        self.clock = None
        self.font = None

        self.sprites_loaded = False
        self.floor_background = None

        self.vacuum = (1, 1)
        self.direction = (0, 1)

        self.walls = set()
        self.dirts = set()

        self.score = 0
        self.total_step = 0

        self.explored_states = []
        self.route = []

        self._setup_window()
        self._load_and_scale_sprites()  

        self.reset()

    def _setup_window(self):
        """Initialize Pygame and the correct render mode."""
        if self.window is None and self.canvas is None:
            pygame.init()
            pygame.font.init()

            font_path = f"{RESOURCES_PATH}font/minecraft/Minecraft.ttf"
            
            try:
                self.font = pygame.font.Font(font_path, 16)
            except FileNotFoundError:
                self.font = pygame.font.SysFont("Arial", 24, bold=True)
            
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))
                pygame.display.set_caption("Vaccum Environment")
                self.clock = pygame.time.Clock()
            else:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                try:
                    self.canvas = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT), pygame.HIDDEN)
                except pygame.error:
                    self.canvas = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))

    def _load_and_scale_sprites(self):
        """Loads assets and resizes them to match the environment's CELL_SIZE."""

        def load_sp(path):
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
            except FileNotFoundError:
                return None
        
        if not self.sprites_loaded:

            """Once defined the structure we can work on this"""
            self.sprites = {}
            components = ["cleaner", "dirt", "floor"]

            for comp in components:
                self.sprites[f"component_{comp}"] = load_sp(os.path.join(RESOURCES_PATH, "components", f"{comp}.png"))

            self.sprites_loaded = True

    def _generate_map(self):
        """Generate a random map of the house to clean"""
        self.walls = set()
        self.dirts = set()

        if self.difficulty == 0:
            # Vertical wall dividing the house into two rooms.
            wall_x = GRID_WIDTH // 2

            # Leave one doorway in the middle.
            doorway_y = GRID_HEIGHT // 2

            for y in range(GRID_HEIGHT):
                if y != doorway_y:
                    self.walls.add((wall_x, y))

            for x in range (GRID_WIDTH):
                for y in range (GRID_HEIGHT):
                    if x == 0 or x == (GRID_WIDTH - 1) or y == 0 or y == (GRID_HEIGHT - 1):
                        self.walls.add((x, y))

            
        elif self.difficulty == 1:
            pass
        elif self.difficulty == 2:
            pass
        else:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.total_step = 0

        self.explored_states = []
        self.route = []

        self._generate_map()

        valid_cells = [(x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) if (x, y) not in self.walls]

        # if difficulty > 0 : also obsttacles to take in account

        if not valid_cells: # debug
            raise RuntimeError(
                "Map generation produced no valid cells."
            )

        vacuum_index = self.np_random.integers(len(valid_cells))

        self.vacuum = valid_cells[vacuum_index]

        self.direction = (0, -1)

        dirt_candidates = valid_cells

        if self.difficulty == 0:
            number_of_dirt = 1

        else:
            number_of_dirt = min(self.n_dirt, len(dirt_candidates))

        if number_of_dirt > 0:
            dirt_indices = self.np_random.choice(len(dirt_candidates), size=number_of_dirt, replace=False)
            self.dirts = {dirt_candidates[int(i)] for i in np.atleast_1d(dirt_indices)}
        else:
            self.dirts = set()

        self._blit_background(reset=True)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def _blit_background(self, reset = True):

        if reset or self.floor_background is None: # Setting for the first time everything
            self.floor_background = pygame.Surface((WIDTH, GAME_HEIGHT))
            # render the house layout (without dirts or anything)
            floor_sprite = self.sprites["component_floor"]
            if floor_sprite is not None:
                for x in range(GRID_HEIGHT):
                    for y in range(GRID_WIDTH):
                        position = (x * CELL_SIZE, y * CELL_SIZE)
                        self.floor_background.blit(self.sprites["component_floor"], position)

                        if floor_sprite is not None:
                            self.floor_background.blit(floor_sprite, position )

                        else:
                            pygame.draw.rect(self.floor_background, (180, 180, 180), (*position, CELL_SIZE, CELL_SIZE))

            for x, y in self.walls:
                position = (x * CELL_SIZE, y * CELL_SIZE)
                pygame.draw.rect(self.floor_background, (50, 50, 50), (*position, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.floor_background, (20, 20, 20), (*position, CELL_SIZE, CELL_SIZE), width=2)   

    def _get_obs(self, done = False):
        if done:
            return self.death_state()
        if self.observation_type == ObservationType.IMAGE:
            frame = self._render_frame()
            if self.render_mode == RenderMode.RGB_ARRAY:
                # rescale pixels in [0, 1]
                frame = frame.astype(np.float32) / 255.0
            return frame
        else:
            # the grid is done such that
            # 0 stands for clean floor
            # 1 stands for dirty floor
            # 2 stands for wall
            # 3 stands for vacuum cleaner
            # other numbers will stand for other things
            grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)

            for dirt in self.dirts:
                y, x = dirt
                grid[y, x] = 1

            for wall in self.walls:
                y, x = wall
                grid[y, x] = 2
            
            y, x = self.vacuum
            grid[y, x] = 3

            return grid

    def _get_info(self):

        return {
            "score": self.score,
            "steps": self.total_step,
            "vacuum_position": self.vacuum,
            "dirt_remaining": len(self.dirts),
            "goal": self.is_goal(),
        }

    def _get_rotation_angle(self, vector):
        """Utility used to determine the rotation of the sprite"""
        mapping = {(0, -1): 0, (-1, 0): 90, (0, 1): 180, (1, 0): 270}
        return mapping.get(vector, 0)

    def is_goal(self):
        """Return True if the entire house is clean."""

        return len(self.dirts) == 0
    
    def get_possible_actions(self, action = None):
        """
        Return the actions that are legal from the
        current vacuum position.

        If action is provided return if it legal or not

        This is useful for search algorithms.
        """
        x, y = self.vacuum
        possible = {
            0: (x, y - 1),  # UP
            1: (x, y + 1),  # DOWN
            2: (x - 1, y),  # LEFT
            3: (x + 1, y),  # RIGHT
            4: (x, y),      # IDLE
            5: (x, y),      # SUCK
        }
        legal = []
        for candidate_action, position in possible.items():
            if candidate_action in (4, 5): # you can always idle and suck
                legal.append(candidate_action)
                continue

            new_x, new_y = position

            if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and (new_x, new_y) not in self.walls):
                legal.append(candidate_action)

        if action is None:
            return legal
        
        return action in legal

    def get_score(self):
        return self.score

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.total_step += 1

        reward = -1
        terminated = False
        truncated = False

        old_position = self.vacuum

        directions = {
            0: (0, -1),  # UP
            1: (0, 1),   # DOWN
            2: (-1, 0),  # LEFT
            3: (1, 0),   # RIGHT
        }

        if action in directions:
            dx, dy = directions[action]

            new_position = (self.vacuum[0] + dx, self.vacuum[1] + dy)

            if self.get_possible_actions(action):
                self.vacuum = new_position
                self.direction = (dx, dy)
                reward = -0.1

            else:
                reward = -3.0

        elif action == 4:
            reward = -0.5

        elif action == 5:
            if self.vacuum in self.dirts:
                self.dirts.remove(self.vacuum)
                reward = 10
                self.score += 10
            else:
                reward = -1

        if self.is_goal():
            terminated = True
            reward += 100
            self.score += 100

        if self.total_step >= self.max_step:
            truncated = True

        self.route.append(self.vacuum)
        _ = old_position

        observation = self._get_obs()
        info = self._get_info()

        return (observation, reward, terminated, truncated, info)
    
    def _render_frame(self):
        """Internal worker function that draws the frame onto the canvas."""

        paint_surface = self.window if self.render_mode == "human" else self.canvas
        paint_surface.fill((40, 40, 40))

        if self.status_bar:
        
            total_seconds = self.total_step // self.metadata["render_fps"]
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            time_str = f"{minutes:02d}.{seconds:02d}"

            diff_text = self.font.render(f"Difficulty {self.difficulty}", True, (255, 255, 255))
            score_text = self.font.render(f"Score {self.score}", True, (255, 215, 0))
            time_text = self.font.render(f"Time {time_str}", True, (255, 255, 255))
            
            text_y = (BAR_HEIGHT - diff_text.get_height()) // 2
            paint_surface.blit(diff_text, (20, text_y))
            paint_surface.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, text_y))
            paint_surface.blit(time_text, (WIDTH - time_text.get_width() - 20, text_y))

        # Floor background + walls
        paint_surface.blit(self.floor_background, (0, BAR_HEIGHT))

        dirt_sprite = self.sprites["component_dirt"]

        # Dirts
        for x, y in self.dirts:
            screen_pos = (x * CELL_SIZE, y * CELL_SIZE + BAR_HEIGHT)
            if dirt_sprite is not None:
                paint_surface.blit(dirt_sprite, screen_pos)
            else:
                pygame.draw.circle(paint_surface, (100, 70, 30), ( x * CELL_SIZE + CELL_SIZE // 2,  y * CELL_SIZE + BAR_HEIGHT + + CELL_SIZE // 2), CELL_SIZE // 4)

        # Vacuum cleaner
        x, y = self.vacuum
        screen_pos = (x * CELL_SIZE, y * CELL_SIZE + BAR_HEIGHT)
        vacuum_sprite = self.sprites["component_cleaner"]
        if vacuum_sprite is not None:
            angle = self._get_rotation_angle(self.direction)
            rotated_vacum = pygame.transform.rotate(vacuum_sprite, angle)
            paint_surface.blit(rotated_vacum, screen_pos)
        else:
            pygame.draw.circle(paint_surface, (40, 100, 220), ( x * CELL_SIZE + CELL_SIZE // 2,  y * CELL_SIZE + BAR_HEIGHT + + CELL_SIZE // 2), CELL_SIZE // 3)

        img_array = pygame.surfarray.array3d(paint_surface)

        # pygame gives:
        #   width x height x channels
        #
        # Gymnasium expects:
        #   height x width x channels

        frame = np.transpose(img_array, (1, 0, 2))

        return frame.astype(np.uint8)
    
    def render(self):
        frame = self._render_frame()

        if self.render_mode == "rgb_array":
            return frame

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return False

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return True

    def close(self):
        if self.window or self.canvas:
            pygame.quit()
            self.window = None
            self.canvas = None
            self.clock = None

if __name__ == '__main__':

    env = VacuumWorld(
        render_mode=RenderMode.HUMAN,
        observation_type=ObservationType.GRID,
        difficulty=0,
    )

    observation, info = env.reset(seed=SEED)

    print("Initial observation:")
    print(observation)

    print()
    print("Initial info:")
    print(info)

    running = True

    while running:

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        running = env.render()

        if terminated or truncated:
            print("Episode finished.")
            print(info)
            running = False
            #observation, info = env.reset(seed=SEED)

    env.close()