import random
import matplotlib.pyplot as plt


class GridWalkSimulator:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.transition_probabilities = self._initialize_probabilities()
        self.sensor_location = self._place_sensor()

    def _initialize_probabilities(self):
        # Создание матрицы перехода вероятностей для каждого узла
        probabilities = {}

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                probabilities[(x, y)] = self._generate_probabilities(x, y)

        return probabilities

    def _generate_probabilities(self, x, y):
        # Генерация вероятностей для одного узла сетки
        moves = []
        if x > 0: moves.append('left')
        if x < self.grid_size - 1: moves.append('right')
        if y > 0: moves.append('up')
        if y < self.grid_size - 1: moves.append('down')

        random.shuffle(moves)
        probs = {direction: 0 for direction in ['left', 'right', 'up', 'down']}
        remaining = 100

        for move in moves[:-1]:
            prob = random.randint(0, remaining)
            probs[move] = prob / 100.0
            remaining -= prob

        if moves:
            probs[moves[-1]] = remaining / 100.0

        return probs

    def _place_sensor(self):
        # Устанавливает датчик в случайное положение на сетке
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

    def simulate_walk(self, start_position):
        # Симуляция одного блуждания от стартовой позиции до датчика
        position = start_position
        steps_taken = 0
        path = [position]

        while position != self.sensor_location:
            steps_taken += 1
            position = self._move_animal(position)
            path.append(position)

        return steps_taken, path

    def _move_animal(self, position):
        # Определяет следующее положение животного на сетке
        x, y = position
        directions = self.transition_probabilities[(x, y)]
        move = random.choices(list(directions.keys()), weights=directions.values())[0]

        if move == 'left':
            x -= 1
        elif move == 'right':
            x += 1
        elif move == 'up':
            y -= 1
        elif move == 'down':
            y += 1

        return (x, y)

    def conduct_experiments(self, num_experiments):
        # Проводит несколько экспериментов и возвращает результаты
        outcomes = []
        example_path = None

        for i in range(num_experiments):
            start = self._generate_start_position()
            steps, path = self.simulate_walk(start)

            if i == 0:
                example_path = path
                # Вывод информации о первой симуляции
                print(f"Датчик расположен в: {self.sensor_location}")
                print(f"Начальная позиция животного: {start}")
                print(f"Количество шагов до датчика: {steps}")

            outcomes.append(steps)

        return outcomes, example_path

    def _generate_start_position(self):
        # Генерация начальной позиции, отличной от позиции датчика
        while True:
            start = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if start != self.sensor_location:
                return start


class ResultVisualizer:
    @staticmethod
    def plot_histogram(attempts):
        # Построение гистограммы попыток
        plt.hist(attempts, bins=20, edgecolor='black')
        plt.title('Гистограмма шагов до достижения датчика')
        plt.xlabel('Количество шагов')
        plt.ylabel('Частота')

    @staticmethod
    def plot_path(grid_size, sensor, path):
        # Визуализация пути одного блуждания
        plt.xlim(-1, grid_size)
        plt.ylim(-1, grid_size)
        plt.grid(True)
        plt.title('Путь случайного блуждания')

        sensor_x, sensor_y = sensor
        plt.plot(sensor_x, sensor_y, 'ro', label='Датчик', markersize=10)

        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, marker='o', color='b', label='Путь')
            plt.plot(path_x[0], path_y[0], 'go', label='Старт')

        plt.legend()

    @staticmethod
    def display(grid_size, attempts, sensor, path):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        ResultVisualizer.plot_histogram(attempts)
        plt.subplot(1, 2, 2)
        ResultVisualizer.plot_path(grid_size, sensor, path)
        plt.tight_layout()
        plt.show()


# Параметры
size_of_grid = 25
number_of_trials = 100

# Инициализация симулятора и проведение экспериментов
simulator = GridWalkSimulator(size_of_grid)
results, path_example = simulator.conduct_experiments(number_of_trials)

# Визуализация результатов
ResultVisualizer.display(size_of_grid, results, simulator.sensor_location, path_example)
