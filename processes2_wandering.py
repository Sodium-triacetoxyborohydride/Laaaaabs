import numpy as np
import matplotlib.pyplot as plt
import random

# Генерация матрицы вероятностей переходов с учетом ограничений
def generate_transition_matrix(size):
    matrix = [[[0] * 4 for _ in range(size)] for _ in range(size)]
    for x in range(size):
        for y in range(size):
            directions = []
            if x > 0:  # Влево
                directions.append(0)
            if x < size - 1:  # Вправо
                directions.append(1)
            if y > 0:  # Вверх
                directions.append(2)
            if y < size - 1:  # Вниз
                directions.append(3)

            probability_sum = 100
            random.shuffle(directions)
            for d in directions[:-1]:
                probability = random.randint(0, probability_sum)
                matrix[x][y][d] = probability / 100.0
                probability_sum -= probability

            # Убедиться, что сумма вероятностей равна 1
            matrix[x][y][directions[-1]] = probability_sum / 100.0

    return matrix

# Выбор случайной позиции, исключая указанную
def select_random_position(size, exclude=None):
    while True:
        pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        if pos != exclude:
            return pos

# Реализация случайного блуждания
def perform_random_walk(size, transition_matrix, start_pos, end_pos):
    steps = 0
    current_pos = start_pos
    path = [current_pos]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # направления: влево, вправо, вверх, вниз

    while current_pos != end_pos:
        i, j = current_pos
        move = random.choices(range(4), weights=transition_matrix[i][j])[0]  # выбор направления на основе вероятностей
        ni, nj = i + directions[move][0], j + directions[move][1]

        # проверка выхода за границы
        if 0 <= ni < size and 0 <= nj < size:
            current_pos = (ni, nj)
            path.append(current_pos)
            steps += 1

    return steps, path

# Проведение эксперимента
def conduct_experiment(size, num_animals):
    transition_matrix = generate_transition_matrix(size)

    end_pos = select_random_position(size)
    steps_list = []

    for _ in range(num_animals):
        start_pos = select_random_position(size, exclude=end_pos)
        steps, path = perform_random_walk(size, transition_matrix, start_pos, end_pos)
        steps_list.append(steps)

    plt.hist(steps_list, bins=30, edgecolor='black')
    plt.title('Histogram of Steps to Reach Sensor')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.show()

    # Визуализация пути первого животного
    if size <= 10:
        plt.figure(figsize=(6, 6))
        for i in range(len(path) - 1):
            plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], 'bo-')
        plt.plot(end_pos[1], end_pos[0], 'ro', label='Sensor')
        plt.plot(path[0][1], path[0][0], 'go', label='Start')
        plt.gca().invert_yaxis()
        plt.xlim(-1, size)
        plt.ylim(-1, size)
        plt.grid(True)
        plt.legend()
        plt.title('Path of an Animal')
        plt.show()

# Запуск эксперимента с заданными параметрами
conduct_experiment(size=25, num_animals=100)
