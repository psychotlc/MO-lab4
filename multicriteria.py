import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from scipy.spatial import distance

CRITERIA_COUNT        = 4

MAIN_CRITERIA_INDEX   = 1
MAIN_CRITERIA_LIMITS  = [0.15, 0, 0.1, 0.25]
SECOND_CRITERIA_INDEX = 0
THIRD_CRITERIA_INDEX  = 2
MAX_WEIGHT            = 10


APPEARANCE_FILL_LIST  = [8/9, 8/3, 8/5, 9/3, 9/5, 3/5]
FINANCE_FILL_LIST     = [7/4, 7/5, 7/8, 4/5, 4/8, 5/8]
HOMELINESS_FILL_LIST  = [6/4, 6/8, 6/7, 4/8, 4/7, 8/7]
CHARACTER_FILL_LIST   = [3/8, 3/5, 3/1, 8/5, 8/1, 5/1]
CRITERIA_FILL_LIST    = [5/6, 5/5, 5/4, 6/5, 6/4, 5/4]

class Multicriteria:

    def __init__(self, params):
        """
        Переопределенный метод __init__
        :params params: хеш таблица
            :params alternative_names: названия альтернатив
            :params criteria_names: названия криетриев
            :params criteria_weight: веса критериев
            :params criterias_direction: направления критериев(max, min)
            :params alternative_matrix: альтернативная матрица
        """
        self.alternative_names   = list(params['alternative_names'])
        self.criteria_names      = list(params['criteria_names'])
        self.criteria_weight     = np.array(params['criterias_weight'])
        self.criterias_directoin = np.array(params['criterias_direction'])
        self.alternative_matrix  = np.array(params['alternative_matrix'], float)

        # нормализуем веса
        self.normalized_weight   = self.norming_vector(self.criteria_weight)

        # нормализуем альтернативную матрицу
        self.normalized_matrix   = self.normalize_matrix(self.alternative_matrix)


    def maximize_alternative_matrix_criteria(self, alternative_matrix, criterias_direction):
        """
        Своидит все криитерии к максимизации.
        :params alternative_matrix: Альтернативная матрица
        :params criterias_direction: направление критерия(max, min)
        """
        for j in range(len(criterias_direction)):
            if criterias_direction[j] == "min":
                for i in range(alternative_matrix.shape[0]):
                    alternative_matrix[i][j] = MAX_WEIGHT - alternative_matrix[i][j] + 1

        return alternative_matrix
    
    def norming_vector(self, vector):
        """
        Метод нормирует вектор
        :params vector: вектор для нормализации
        """

        normalize_weight = vector.copy()
        weight_sum = np.sum(normalize_weight)

        normalize_weight = normalize_weight / weight_sum

        return normalize_weight
    
    def normalize_matrix(self, matrix):
        """
        Метод нормализует матрицу. Не трогай столбец j=MAIN_CRITERIA_INDEX
        :params matrix: Матрица для нормализации
        """

        normalized_matrix = matrix.copy()
        minimums = normalized_matrix.min(axis=0)
        maximums = normalized_matrix.max(axis=0)

        for j in range(normalized_matrix.shape[1]):
            if j != MAIN_CRITERIA_INDEX:
                for i in range(matrix.shape[0]):
                    normalized_matrix[i][j] = (normalized_matrix[i][j] - minimums[j]) / (
                            maximums[j] - minimums[j])                                       ## (A[i,j]-A_min[j])/(A_max[j]-A_min[j])

        return normalized_matrix
    
    def out_matrix(self, matrix):
        """
        Выводит матрицу альтернатив.
        :params matrix: матрица альтернатив
        """
        table = PrettyTable()

        table.field_names = ["Альтернативы"] + self.criteria_names

        for i in range(len(self.alternative_names)):
            new_row = [self.alternative_names[i]]
            for j in range(len(self.criteria_names)):
                new_row.append(round(matrix[i][j], 2))

            table.add_row(new_row)

        return table

    def out_weight(self):
        """
        Выводит вектор весов критериев.
        """
        out = "Составляем веткор весов критериев, используя шкалу 1-10:\n"
        table = PrettyTable()
        table.field_names = self.criteria_names
        table.add_row(self.criteria_weight)

        out += table.__str__()

        out += "\nНормализовав, получим вектор " + self.normalized_weight.__str__()

        return out
    
    def main_criteria_method(self):
        """
        Решение методом главного критерия.
        """
        print("\n1) Метод замены критериев ограничениями (метод главного критерия).\n"
                "Составим матрицу оценок альтернатив.")
        print(self.out_matrix(self.alternative_matrix))

        matrix = self.normalized_matrix.copy()
        maximums = matrix.max(axis=0)

        print("Ограничения:")
        for j in range(len(self.criteria_names)):
            if j != MAIN_CRITERIA_INDEX:
                print(f"{self.criteria_names[j]} не менее {MAIN_CRITERIA_LIMITS[j] * maximums[j]}")

        print(f"\nПроведём нормирование матрицы:\n{self.out_matrix(self.normalize_matrix(self.alternative_matrix))}")

        constraints = []
        for j in range(len(self.criteria_names)):
            if j == MAIN_CRITERIA_INDEX:
                constraints.append(None)
            else:
                constraints.append(MAIN_CRITERIA_LIMITS[j] * maximums[j])

        acceptable_rows = []

        for i in range(len(self.alternative_names)):
            row = matrix[i]
            if (row < MAIN_CRITERIA_LIMITS).any():
                continue

            acceptable_rows.append(i)

        if len(acceptable_rows):
            print("При заданных ограничениях приемлимыми являются следующие решения:")
            for i in acceptable_rows:
                print(self.alternative_names[i])

            max_alternative = None
            for i in acceptable_rows:
                curr = self.normalized_matrix[i][MAIN_CRITERIA_INDEX]
                if max_alternative is None or self.normalized_matrix[max_alternative][MAIN_CRITERIA_INDEX] < curr:
                    max_alternative = i

            print("Итоговое решение:")
            print(self.alternative_names[max_alternative])

        else:
            print("При заданных ограничениях не нашлось приемлимых решений.")

    def pareto_method(self):
        """
        Решение формированием и сужением множества Парето.
        """

        print(
                f"\n 2) Формирование и сужение множества Парето. \n"
                f"Выберем в качестве критериев для данного метода {self.criteria_names[SECOND_CRITERIA_INDEX]} и "
                f"{self.criteria_names[THIRD_CRITERIA_INDEX]}.\n"
                f"{self.criteria_names[SECOND_CRITERIA_INDEX]} - по оси X, "
                f"{self.criteria_names[THIRD_CRITERIA_INDEX]} - по оси Y.\n"
                f"Сформируем множество Парето графическим методом. (см. график)"
        )

        plt.title("Графическое решение методом сужения множества Парето.")
        plt.xlabel(f"Критерий: {self.criteria_names[SECOND_CRITERIA_INDEX]}")
        plt.ylabel(f"Критерий: {self.criteria_names[THIRD_CRITERIA_INDEX]}")

        xValues = self.alternative_matrix[:, SECOND_CRITERIA_INDEX]
        yValues = self.alternative_matrix[:, THIRD_CRITERIA_INDEX]
        plt.grid()
        plt.plot(xValues, yValues, "b")

        euclid_length = []
        for i in range(len(self.alternative_matrix[:, SECOND_CRITERIA_INDEX])):
            x_i = self.alternative_matrix[i, SECOND_CRITERIA_INDEX]
            y_i = self.alternative_matrix[i, THIRD_CRITERIA_INDEX]
            plt.plot(x_i, y_i, "bo")
            plt.text(x_i + 0.1, y_i, self.alternative_names[i][0])

            euclid_distance = distance.euclidean((x_i, y_i), (xValues.max(), yValues.max()))

            euclid_length.append(euclid_distance)

        plt.plot(xValues.max(), yValues.max(), "rD")
        plt.text(xValues.max() + 0.1, yValues.max() + 0.1, "Точка утопии")

        plt.show()

        min_index = min(enumerate(euclid_length), key=lambda x: x[1])[0]

        print(
            f"Исходя из графика можно сказать, что Евклидово расстояние до "
            f"точки минимально для варианта:\n{self.alternative_names[min_index]}"
        )

    def normalize_by_columns(self, current_matrix):
        """
        Нормализует колонки в матрице.
        """
        matrix = current_matrix.copy()
        for i in range(len(self.criteria_names)):
            col_sum = np.sum(matrix[i])
            matrix[i] = matrix[i] / col_sum

        return matrix

    def criteria_evaluation(self, y12, y13, y14, y23, y24, y34):
        """
        Составляет матрицу экспертных оценок
        :params yij: оценка критериев
        """
        table = PrettyTable()
        table.field_names = [""] + self.criteria_names
        table.add_row([self.criteria_names[0]] + [0, y12, y13, y14])
        table.add_row([self.criteria_names[1]] + [1 - y12, 0, y23, y24])
        table.add_row([self.criteria_names[2]] + [1 - y13, 1 - y23, 0, y34])
        table.add_row([self.criteria_names[3]] + [1 - y14, 1 - y24, 1 - y34, 0])

        return table
    
    def weight_and_combined_method(self):
        """
        Решение методом взвешивания и объединения критериев.
        """
        alternative_max_criteria_matrix = self.maximize_alternative_matrix_criteria(self.alternative_matrix, self.criterias_directoin)

        rating_matrix = self.normalize_by_columns(alternative_max_criteria_matrix)
        rm = self.normalize_by_columns(self.alternative_matrix)

        print(  "\n 3) Взвешивание и объединение критериев. \n"
                f"Составим матрицу рейтингов альтернатив по критериям, используя шкалу 1-10: \n\n "
                f"{self.out_matrix(self.alternative_matrix)} \n\n Нормализуем её: \n"
                f"{self.out_matrix(rm)}\n")

        print("Составим экспертную оценку критериев (по методу попарного сравнения):\n")
        y12 = 0.5
        y13 = 1
        y14 = 1
        y23 = 1
        y24 = 1
        y34 = 1
        print(self.criteria_evaluation(y12, y13, y14, y23, y24, y34))

        weight_vector = np.array([y12 + y13 + y14, y12 + y14, y14 + y24 + y34, 0])

        weight_vector = self.norming_vector(weight_vector)

        print(f"alpha = {weight_vector}")

        weight_vector.transpose()

        combine_criteria = rating_matrix.dot(weight_vector)

        print(  f"Умножив нормализированную матрицу на нормализированный вектор весов критериев, "
                f"получаем значения объединённого критерия альтернатив:\n{combine_criteria}")

        max_index = max(range(len(combine_criteria)), key=combine_criteria.__getitem__)

        print(f"Наиболее приемлемой является альтернатива:\n{self.alternative_names[max_index]}")

    def pair_compare_matrix(self, fill_list):
        """
        Заполянет матрицу попарных сравнений.
        :params fill_list: массив из матриц попарных оценок для каждого критерияя
        """
        k = 0
        pc_matrix = np.ones((CRITERIA_COUNT, CRITERIA_COUNT))
        # Заполняем верхний треугольник.
        for i in range(CRITERIA_COUNT):
            for j in range(CRITERIA_COUNT):
                if i < j:
                    pc_matrix[i][j] = round(fill_list[k], 3)
                    k += 1

        k = 0
        # Заполняем нижний треугольник.
        for i in range(CRITERIA_COUNT):
            for j in range(CRITERIA_COUNT):
                if i < j:
                    pc_matrix[j][i] = round(1 / fill_list[k], 3)
                    k += 1

        return pc_matrix

    def pair_compare_table(self, names, main_matrix, sum_col, normalize_sum_col):
        """
        Составляет таблицу с матрицей попарных сравнений
        :params names: заголовки столбцов
        :params main_matrix: матрица сравнения критериев
        :params sum_col: сумма строки
        :normalize_sum_col: сумма нормализованной строки
        """
        table = PrettyTable()
        table.field_names = [""] + names + ["Сумма по строке", "Нормированная сумма по строке"]
        for i in range(len(self.alternative_names)):
            row = [names[i]] + list(main_matrix[i])
            row.append(round(sum_col[i], 2))
            row.append(round(normalize_sum_col[i], 2))
            table.add_row(row)

        return table


    def hierarchies_analysis_method(self):
        """
        Решение методом анализа иерархий.
        """
        print(  "\n4) Меотд анализа иерархий.\nСоставим для каждого из критериев матрицу попарного сравнения альтернатив,"
                " нормализуем ее и матрицу из векторов приоритетов альтернатив:\n")

        fill_lists = [APPEARANCE_FILL_LIST, FINANCE_FILL_LIST, HOMELINESS_FILL_LIST, CHARACTER_FILL_LIST]

        hierarchies_matrix = None

        for i in range(len(self.criteria_names)):
            print(f"• {self.criteria_names[i]}")
            main_matrix = self.pair_compare_matrix(fill_lists[i])

            sum_col = np.sum(main_matrix, axis=1)

            normalize_sum_col = self.norming_vector(sum_col)
            print(self.pair_compare_table(self.alternative_names, main_matrix, sum_col, normalize_sum_col))

            if hierarchies_matrix is None:
                hierarchies_matrix = normalize_sum_col.transpose()
            else:
                hierarchies_matrix = np.c_[hierarchies_matrix, normalize_sum_col.transpose()]

        print("Оценка приоритетов:")
        criteria_matrix = self.pair_compare_matrix(CRITERIA_FILL_LIST)
        sum_col = np.sum(criteria_matrix, axis=1)

        normalize_sum_col = self.norming_vector(sum_col)
        print(self.pair_compare_table(self.criteria_names, criteria_matrix, sum_col, normalize_sum_col))

        normalize_sum_col.transpose()

        resulted_vec = hierarchies_matrix.dot(normalize_sum_col.transpose())

        

        print(  "Умножив матрицу, состваленную из норм. сумм по строкам на вектор-столбец оценки приоритетов, "
                "получим вектор:")
        
        print(resulted_vec)

        max_index = np.argmax(resulted_vec)

        print(f"Наиболее приемлемой является альтернатива:\n{self.alternative_names[max_index]}")