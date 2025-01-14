import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, value, PULP_CBC_CMD, LpMinimize

def show_res(res, data, proteins, fats, carbs, prices, calories):
    sum_prot = 0
    sum_fat = 0
    sum_carb = 0
    sum_price = 0
    sum_calories = 0
    print('\n', '-' * 20, 'РЕЗУЛЬТАТ', '-' * 20)
    print("\nРекомендуемый набор блюд:\n")
    for i in range(len(res)):
        if res[i] > 0:
            number = int(i)
            colvo = res[i]
            print(str(i + 1) + '.', data[number][0], round(colvo, 2), 'шт.')
            sum_prot += proteins[number] * colvo
            sum_fat += fats[number] * colvo
            sum_carb += carbs[number] * colvo
            sum_price += prices[number] * colvo
            sum_calories += calories[number] * colvo

    print('\nСтоимость набора:', round(sum_price, 2), 'руб.')
    print('Белки:', round(sum_prot, 2), 'г.')
    print('Жиры:', round(sum_fat, 2), 'г.')
    print('Углеводы:', round(sum_carb, 2), 'г.')
    print('Калории:', round(sum_calories, 2))

def main():
    file_path = 'menu.txt'
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()

    # Пропускаем заголовок
    data = [line.strip().split(',') for line in lines[1:]]

    # Извлечение данных и преобразование в числа
    names = [item[0] for item in data]
    calories = np.array([int(item[1]) for item in data])
    proteins = np.array([int(item[2]) for item in data])
    fats = np.array([int(item[3]) for item in data])
    carbs = np.array([int(item[4]) for item in data])
    prices = np.array([int(item[5]) for item in data])

    x, b, signs = [], [], []

    # проверка ввода y/n
    def get_valid_input(prompt, valid_options):
        while True:
            user_input = input(prompt).strip().lower()
            if user_input in valid_options:
                return user_input
            else:
                print(f"Пожалуйста, введите {' или '.join(valid_options)}.")

    # проверка ввода чисел для КБЖУ
    def get_valid_number(prompt):
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Пожалуйста, введите числовое значение.")

    # считываем ограничения от пользователя
    use_calories = get_valid_input("Использовать ограничение по калориям? (y/n): ", ['y', 'n']) == 'y'
    use_proteins = get_valid_input("Использовать ограничение по белкам? (y/n): ", ['y', 'n']) == 'y'
    use_fats = get_valid_input("Использовать ограничение по жирам? (y/n): ", ['y', 'n']) == 'y'
    use_carbs = get_valid_input("Использовать ограничение по углеводам? (y/n): ", ['y', 'n']) == 'y'
    percent = int(input("Введите процент : ")) / 100
    x_max = int(input("Введите максимальное количество одного блюда в рационе: "))

    if use_calories:
        D = get_valid_number("Введите количество калорий: ")
        deltaD = D * percent
        x.append(calories)
        x.append(calories)
        b.append(D - deltaD)
        b.append(D + deltaD)
        signs.append('>=')
        signs.append('<=')

    if use_proteins:
        A = get_valid_number("Введите количество белков: ")
        deltaA = A * percent
        x.append(proteins)
        x.append(proteins)
        b.append(A - deltaA)
        b.append(A + deltaA)
        signs.append('>=')
        signs.append('<=')

    if use_fats:
        B = get_valid_number("Введите количество жиров: ")
        deltaB = B * percent
        x.append(fats)
        x.append(fats)
        b.append(B - deltaB)
        b.append(B + deltaB)
        signs.append('>=')
        signs.append('<=')

    if use_carbs:
        C = get_valid_number("Введите количество углеводов: ")
        deltaC = C * percent
        x.append(carbs)
        x.append(carbs)
        b.append(C - deltaC)
        b.append(C + deltaC)
        signs.append('>=')
        signs.append('<=')

    # Создание задачи линейного программирования
    prob = LpProblem("Diet Problem", LpMinimize)

    # Определение переменных
    food_vars = LpVariable.dicts("Food", list(range(len(prices))), lowBound=0, upBound=x_max, cat=LpInteger)

    # Целевая функция
    prob += lpSum([prices[i] * food_vars[i] for i in range(len(prices))]), "Total Cost"

    # Добавление ограничений
    for i, coefs in enumerate(x):
        if signs[i] == '>=':
            prob += lpSum([coefs[j] * food_vars[j] for j in range(len(prices))]) >= b[i]
        elif signs[i] == '<=':
            prob += lpSum([coefs[j] * food_vars[j] for j in range(len(prices))]) <= b[i]

    # Решение задачи
    prob.solve()

    # Получение и вывод результатов
    result = [value(food_vars[i]) for i in range(len(prices))]
    show_res(result, data, proteins, fats, carbs, prices, calories)

def main1():
    # Целевая функция: максимизация z = -5x1 - 3x2 - 2x3
    z = [5, 3, 2]

    # Коэффициенты для ограничений
    x1 = [2, 3, 1]  # для первого ограничения
    x2 = [4, 1, 2]  # для второго ограничения
    x3 = [1, 2, 3]  # для третьего ограничения
    x = [x1, x2, x3]

    # Правая часть ограничений
    b = [10, 15, 12]

    # Знаки ограничений
    signs = ['<=', '<=', '<=']

    # Создание задачи линейного программирования (максимизация)
    prob = LpProblem("Integer Linear Programming Example", LpMaximize)

    # Определение переменных с ограничением целочисленности
    vars = LpVariable.dicts("Var", list(range(len(z))), lowBound=0, cat=LpInteger)

    # Определяем целевую функцию
    prob += lpSum([z[i] * vars[i] for i in range(len(z))]), "Total Profit"

    # Добавление ограничений
    for i, coefs in enumerate(x):
        if signs[i] == '<=':
            prob += lpSum([coefs[j] * vars[j] for j in range(len(z))]) <= b[i], f"Constraint_{i}_le"
        elif signs[i] == '>=':
            prob += lpSum([coefs[j] * vars[j] for j in range(len(z))]) >= b[i], f"Constraint_{i}_ge"

    # Решение задачи
    prob.solve(PULP_CBC_CMD(msg=1))

    # Получение и вывод результатов
    result = [value(vars[i]) for i in range(len(z))]

    # Печать результатов
    print('Результаты:')
    for i in range(len(result)):
        print(f"Количество x{i + 1}: {result[i]}")
    print("Оптимальное значение целевой функции:", value(prob.objective))

def main2():
    # Целевая функция: минимизация z = 2x1 + 3x2 + 4x3
    z = [2, 3, 4]

    # Коэффициенты для ограничений
    x1 = [1, 2, 1]  # для первого ограничения
    x2 = [2, 1, 3]  # для второго ограничения
    x3 = [1, 3, 2]  # для третьего ограничения
    x = [x1, x2, x3]

    # Правая часть ограничений
    b = [8, 10, 12]

    # Знаки ограничений
    signs = ['>=', '>=', '>=']

    # Создание задачи линейного программирования
    prob = LpProblem("Integer Linear Programming Example", LpMinimize)

    # Определение переменных с ограничением целочисленности
    vars = LpVariable.dicts("Var", list(range(len(z))), lowBound=0, cat=LpInteger)

    # Определяем целевую функцию
    prob += lpSum([z[i] * vars[i] for i in range(len(z))]), "Total Cost"

    # Добавление ограничений
    for i, coefs in enumerate(x):
        if signs[i] == '>=':
            prob += lpSum([coefs[j] * vars[j] for j in range(len(z))]) >= b[i], f"Constraint_{i}_ge"
        elif signs[i] == '<=':
            prob += lpSum([coefs[j] * vars[j] for j in range(len(z))]) <= b[i], f"Constraint_{i}_le"

    # Решение задачи
    prob.solve(PULP_CBC_CMD(msg=1))

    # Получение и вывод результатов
    result = [value(vars[i]) for i in range(len(z))]

    # Печать результатов
    print('Результаты:')
    for i in range(len(result)):
        print(f"Количество x{i + 1}: {result[i]}")
    print("Оптимальное значение целевой функции:", value(prob.objective))

main()