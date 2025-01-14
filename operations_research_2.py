import numpy as np
import copy

def sum_raw(raw1, raw2):
    ans = []
    for i in range(len(raw1)):
        ans.append(raw1[i] + raw2[i])
    return ans


def vect_mult_number(v, a):
    return [el * a for el in v]



def is_optimal(symplex_matrix, factor):
    last_row = symplex_matrix[-1, :-1]
    if factor == 'min':
        return np.all(last_row <= 0)
    else:
        return np.all(last_row <= 0)


def find_basis(symplex_matrix):
    num_rows, num_cols = symplex_matrix.shape
    basis_indices = []
    for col_index in range(num_cols - 1):
        col = symplex_matrix[:, col_index]
        if np.sum(col == 1) == 1 and np.sum(col == 0) == num_rows - 1:
            basis_indices.append(col_index)
    return basis_indices


def rect_rule(solve_element, solve_column_ind, solve_raw_ind, symplex_matrix):
    r = int(solve_raw_ind)
    s = int(solve_column_ind)
    for i in range(len(symplex_matrix)):
        if i != r:
            f = symplex_matrix[i][s] / solve_element
            for j in range(len(symplex_matrix[i])):
                symplex_matrix[i][j] -= symplex_matrix[r][j] * f
    return symplex_matrix


def show_res(res,data,proteins,fats,carbs,prices,calories):
    sum_prot = 0
    sum_fat = 0
    sum_carb = 0
    sum_price = 0
    sum_calories=0
    print('\n', '-' * 20, 'РЕЗУЛЬТАТ', '-' * 20)
    print("\nРекомендуемый набор блюд:\n")
    for i in range(len(res)):
        number = int(res[i][0]) - 1
        colvo = res[i][1]
        print(str(i+1)+'.',data[number][0], round(colvo,2),'шт.')
        sum_prot = sum_prot + proteins[number] * colvo
        sum_fat = sum_fat + fats[number] * colvo
        sum_carb = sum_carb + carbs[number] * colvo
        sum_price = sum_price + prices[number] * colvo
        sum_calories=sum_calories+calories[number]*colvo

    print('\nCтоимость набора:', round(sum_price,2), 'руб.')
    print('Белки:', round(sum_prot,2), 'г.')
    print('Жиры:', round(sum_fat,2), 'г.')
    print('Углеводы:', round(sum_carb,2), 'г.')
    print('Калории:', round(sum_calories,2))

def symplex_method(symplex_matrix, var_names, row_base_vars, factor, show_solving):
    f = show_solving
    if factor == 'max':
        print('\n', '\n', '-' * 10, 'МАКСИМИЗАЦИЯ', '\n', '-' * 10, '\n')
    elif factor == 'min':
        print('\n', '\n', '-' * 20, 'МИНИМИЗАЦИЯ', '-' * 20, '\n')
    else:
        print('ОШИБКА', '\n')
        return 0

    iteration = 0
    max_iterations = 1000

    while not is_optimal(symplex_matrix, factor):
        if f:
            print('z-строка:\n', symplex_matrix[-1, :-1])
        if iteration > max_iterations:
            print("Превышено максимальное количество итераций. Возможное зацикливание.")
            break

        z_vect = symplex_matrix[-1, :-1]

        # Выбор разрешающего столбца
        if factor == 'min':
            solve_column_ind = np.argmax(z_vect)
        else:
            solve_column_ind = np.argmax(z_vect)

        if f:
            print('Решающий столбец: ', solve_column_ind)

        min_ratio = float('inf')
        solve_raw_ind = -1
        for i in range(len(symplex_matrix) - 1):
            if symplex_matrix[i][solve_column_ind] > 0:
                ratio = symplex_matrix[i][-1] / symplex_matrix[i][solve_column_ind]
                if abs(ratio) < min_ratio:
                    min_ratio = abs(ratio)
                    solve_raw_ind = i

        if solve_raw_ind == -1:
            print('Задача не имеет ограниченного решения.')
            return

        if f:
            print('Решающая строка', solve_raw_ind)

        solve_element = symplex_matrix[solve_raw_ind][solve_column_ind]

        if f:
            print('Решающий элемент: ', solve_element)

        row_base_vars[solve_raw_ind] = var_names[solve_column_ind]


        symplex_matrix = rect_rule(solve_element, solve_column_ind, solve_raw_ind, symplex_matrix)
        symplex_matrix[solve_raw_ind] /= solve_element
        if f:
            print("Текущая симплекс-матрица:\n", symplex_matrix)
            print("Текущий базис:\n", row_base_vars)

    # Вывод значений переменных и целевой функции
    solution = symplex_matrix[:, -1]
    z_value = symplex_matrix[-1, -1]

    z_ans =z_value
    if f:
        print('\n', '-' * 20, 'РЕЗУЛЬТАТ', '-' * 20)
        print('\n', "\nТекущий базис:", row_base_vars)
        print("Оптимальное решение:", solution[:-1])
        if factor=='min':
            z_ans = z_value
            print("Значение целевой функции (z):", z_value)
        else:
            z_ans = -z_value
            print("Значение целевой функции (z):", -z_value)

    res = []
    for i in range(len(row_base_vars)):
        if row_base_vars[i][0] == 'x':
            res.append([row_base_vars[i][1:], float(solution[i])])
    return res , z_ans


def M_method(z, x, b, signs, M, factor, show_solving):
    f = show_solving
    limit_num = len(x)

    A = np.array(x)
    B = np.array(b)

    # Обработка знаков неравенств и создание дополнительных столбцов
    additional_cols = []
    for i in range(limit_num):
        extra_col = np.zeros(limit_num)
        if signs[i] == '<=':
            extra_col[i] = 1
            additional_cols.append(extra_col)
        elif signs[i] == '>=':
            extra_col[i] = -1
            additional_cols.append(extra_col)

    # Добавление дополнительных столбцов S в матрицу A
    if additional_cols:
        A = np.hstack((A, np.array(additional_cols).T))

    # Создание искусственной единичной матрицы
    artificial_matrix = np.eye(limit_num)

    symplex_matrix = np.hstack((A, artificial_matrix.T, B.reshape(-1, 1)))

    # Преобразование вектора коэффициентов целевой функции
    z_vect = np.hstack(
        (vect_mult_number(z, -1), [0] * len(additional_cols), np.ones(artificial_matrix.shape[0]) * (-M), [0]))

    symplex_matrix = np.vstack((symplex_matrix, z_vect))

    if f:
        print("Начальная симплекс-матрица:")
        print(symplex_matrix)

    for i in range(limit_num):
        symplex_matrix[limit_num] = sum_raw(symplex_matrix[limit_num], symplex_matrix[i] * (M))
    if f:
        print(symplex_matrix)

    var_names = [f"x{i + 1}" for i in range(len(z))] + [f"s{i + 1}" for i in
                                                        range(len(additional_cols))] + [f"r{i + 1}" for i in range(
        artificial_matrix.shape[0])] + ["b"]
    row_base_vars = []
    indexes = find_basis(symplex_matrix)
    for i in range(len(indexes)):
        row_base_vars.append(var_names[indexes[i]])

    # Выполнение симплекс-метода
    return symplex_method(symplex_matrix, var_names, row_base_vars, factor, show_solving)

def BranchBoundsMethod(res, z, x, b, signs, M, m, it, best_solution=None, best_z=None, factor='min', iterations_limit=10, eps=1e-6):
    if it >= iterations_limit:
        print('Превышено максимальное количество итераций')
        return best_solution, best_z

    # Проверка на целочисленность
    for el in res:
        index = int(el[0]) - 1
        value = el[1]

        if not np.isclose(value, round(value), atol=eps):  # Если значение не является целым
            it += 1

            # Создание ограничений для округления вниз
            x_copy1 = copy.deepcopy(x)
            signs_copy1 = copy.deepcopy(signs)
            b_copy1 = copy.deepcopy(b)

            floor_constraint = np.zeros_like(z)
            floor_constraint[index] = 1
            x_copy1.append(floor_constraint)
            signs_copy1.append('<=')
            b_copy1.append(np.floor(value))

            print(f"Добавлено ограничение для округления вниз: x{index + 1} <= {np.floor(value)}")

            try:
                new_res, z_ans = M_method(z, x_copy1, b_copy1, signs_copy1, M, factor, False)
                new_res = [[el[0], round(el[1], 5)] for el in new_res]  # Округление результата до 5 знаков
                print(new_res)
                print(round(z_ans, 5))
                if factor == 'max' and check_int(new_res) and (best_z is None or abs(z_ans) > abs(best_z)):
                    best_z = z_ans
                    best_solution = new_res
                elif factor == 'min' and check_int(new_res) and (best_z is None or abs(z_ans) < abs(best_z)):
                    best_z = z_ans
                    best_solution = new_res
                if (best_z is not None) and (z_ans > best_z):
                    print("Нет целочисленного решения")
                    break
            except Exception as e:
                print(f"Ошибка при решении подзадачи: {e}")

            # Рекурсивный вызов для поиска целого решения
            best_solution, best_z = BranchBoundsMethod(new_res, z, x_copy1, b_copy1, signs_copy1, M, m, it,
                                                       best_solution, best_z, factor, iterations_limit, eps)

            # Создание ограничений для округления вверх
            x_copy2 = copy.deepcopy(x)
            signs_copy2 = copy.deepcopy(signs)
            b_copy2 = copy.deepcopy(b)

            ceil_constraint = np.zeros_like(z)
            ceil_constraint[index] = 1
            x_copy2.append(ceil_constraint)
            signs_copy2.append('>=')
            b_copy2.append(np.ceil(value))

            print(f"Добавлено ограничение для округления вверх: x{index + 1} >= {np.ceil(value)}")

            try:
                new_res, z_ans = M_method(z, x_copy2, b_copy2, signs_copy2, M, factor, False)
                new_res = [[el[0], round(el[1], 5)] for el in new_res]  # Округление результата до 5 знаков
                print(new_res)
                print(round(z_ans, 5))
                if factor == 'max' and check_int(new_res) and (best_z is None or abs(z_ans) < abs(best_z)):
                    best_z = z_ans
                    best_solution = new_res
                elif factor == 'min' and check_int(new_res) and (best_z is None or abs(z_ans) < abs(best_z)):
                    best_z = z_ans
                    best_solution = new_res
                if (best_z is not None) and (z_ans > best_z):
                    print("Нет целочисленного решения")
                    break
            except Exception as e:
                print(f"Ошибка при решении подзадачи: {e}")

            # Рекурсивный вызов для поиска целого решения
            best_solution, best_z = BranchBoundsMethod(new_res, z, x_copy2, b_copy2, signs_copy2, M, m, it,
                                                       best_solution, best_z, factor, iterations_limit, eps)

            break  # Выходим из цикла после обработки переменной

    # Округление финального лучшего результата
    if best_solution is not None:
        best_solution = [[el[0], round(el[1], 5)] for el in best_solution]
    if best_z is not None:
        best_z = round(best_z, 5)

    return best_solution, best_z

def print_res_mini(res,m):
  x_res=[0]*m
  for el in res:
    x_res[int(el[0])-1]=el[1]
  for i in range (len(x_res)):
    print('x'+str(i+1),' = ',x_res[i])

def check_int(res):
    for el in res:
        if int(el[1]) != el[1]:
            return False
    return True

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

    # заносим ограничения в списки
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

    # запрашиваем максимальное количество раз, которое каждое блюдо может встречаться
    max_dish_count = get_valid_number(
        "Введите максимальное количество раз, которое каждое блюдо может встречаться (например, 1 или 2): ")
    for i, item in enumerate(data):
        if max_dish_count > 0:
            product_constraint = np.zeros(len(data))
            product_constraint[i] = 1
            x.append(product_constraint)
            b.append(max_dish_count)
            signs.append('<=')

    # целевая функция: минимизация стоимости
    z = prices
    M = 1000

    # Начальное решение
    res, z_ans = M_method(z, x, b, signs, M, 'min', False)
    print("Начальное приближенное решение:")
    print(res)
    print("Начальное значение целевой функции:", z_ans)

    # Применение метода ветвей и границ для нахождения целого решения
    it = 0
    best_solution, best_z = BranchBoundsMethod(res, z, x, b, signs, M, len(z), it, factor='min')

    # Вывод результатов
    if best_solution:
        print("Оптимальное целочисленное решение:")
        show_res(best_solution, data, proteins, fats, carbs, prices, calories)
        print("Оптимальное значение целевой функции:", best_z)
    else:
        print("Целочисленное решение не найдено.")

#------------------------------------------------------------------------------
# для проверки
def main1():
    m = 3
    z = [-5, -3, -2]
    x1 = [2, 3, 1]
    x2 = [4, 1, 2]
    x3 = [1, 2, 3]
    x = [x1, x2, x3]
    b = [10, 15, 12]
    signs = ['<=', '<=', '<=']

    M = 10
    res, z_ans = M_method(z, x, b, signs, M, 'max', False)
    print(res)
    print(-1*z_ans)
    print_res_mini(res, m)
    it = 0
    best_solution, best_z = BranchBoundsMethod(res, z, x, b, signs, M, m, it,'max')


    if best_solution is not None:
        print("Оптимальное целочисленное решение:", best_solution)
        print("Оптимальное значение целевой функции:", best_z)
    else:
        print("Целочисленное решение не найдено.")


def main2():
    m = 3
    z = [2, 3, 4]
    x1 = [1, 2, 1]
    x2 = [2, 1, 3]
    x3 = [1, 3, 2]
    x = [x1, x2, x3]
    b = [8, 10, 12]
    signs = ['>=', '>=', '>=']

    M = 10
    res, z_ans = M_method(z, x, b, signs, M, 'min', False)
    print(res)
    print(z_ans)
    print_res_mini(res, m)
    it = 0
    best_solution, best_z = BranchBoundsMethod(res, z, x, b, signs, M, m, it, 'min')

    if best_solution is not None:
        print("Оптимальное целочисленное решение:", best_solution)
        print("Оптимальное значение целевой функции:", best_z)
    else:
        print("Целочисленное решение не найдено.")

main()