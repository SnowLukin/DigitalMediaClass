1. установить размер матрицы свертки и среднее квадратичное
отклонение;
2. заполнить матрицу свертки значениями функции Гаусса с мат.
ожиданием, равным координатам центра матрицы;
3. нормировать матрицу так, чтобы сумма элементов равнялась 1;
4. создать копию изображения;
5.  для каждого внутреннего пикселя копии рассчитать новое значение
насыщенности пикселя по формуле (1) и ЗАПИСАТЬ это значение в пиксель
нового изображения.

# Размытие по Гауссу с использованием OpenCV
### Этот скрипт Python демонстрирует использование OpenCV для применения размытия по Гауссу к изображению. Скрипт также сравнивает результаты размытия по Гауссу, примененного с помощью OpenCV, с результатами пользовательской функции размытия по Гауссу.

# Импорт необходимых библиотек
import cv2
import numpy as np
В этом скрипте мы импортируем библиотеку OpenCV ( cv2) и библиотеку NumPy ( np).

# Гауссова функция
```python
def gauss(x, y, omega, a, b):
    omegaIn2 = 2 * omega ** 2
    m1 = 1/(np.pi * omegaIn2)
    m2 = np.exp(-((x-a) ** 2 + (y-b) ** 2)/omegaIn2)
    return m1*m2
```
Эта функция вычисляет значение функции Гаусса для заданных x, y, omega, aи b. Функция Гаусса — это распространенный способ представления точечного источника или импульса, который часто используется при обработке изображений.

# Функция размытия по Гауссу
```python
def gaussBlur(img, size, deviation):
    kernel = np.ones((size, size))
    a = b = (size+1) // 2

    for i in range(size):
        for j in range(size):
            kernel[i, j] = gauss(i, j, deviation, a, b)

    sum = 0
    for i in range(size):
        for j in range(size):
            sum += kernel[i, j]
    for i in range(size):
        for j in range(size):
            kernel[i, j] /= sum

    blur = img.copy()
    sx = size // 2
    sy = size // 2
    for i in range(sx, blur.shape[0]-sx):
        for j in range(sy, blur.shape[1]-sy):
            value = 0
            for k in range(-(size//2), size//2+1):
                for l in range(-(size//2), size//2+1):
                    value += img[i + k, j + l] * kernel[(size//2) + k,  (size//2) + l]
            blur[i, j] = value

    return blur
```

Эта функция применяет размытие по Гауссу к входному изображению, `img` используя ядро  размера `size` и стандартного отклонения `deviation`. Он вычисляет ядро Гаусса, вызывая `gauss` функцию для каждого пикселя в ядре. Затем ядро нормализуется путем деления каждого элемента на сумму всех элементов. Затем к изображению применяется размытие по Гауссу путем свертки его с ядром.

# Основная функция
```python
def lr3():
    img = cv2.imread("kot2.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img', img)

    deviation = 5
    size = 3
    blur1 = gaussBlur(img, size, deviation)
    cv2.imshow(f'size: {size}, deviation: {deviation}', blur1)

    deviation = 10
    size = 7
    blur2 = gaussBlur(img, size, deviation)
    cv2.imshow(f'size: {size}, deviation: {deviation}', blur2)

    blurOpenCV = cv2.GaussianBlur(img, (size, size), deviation)
        
    cv2.imshow(f'OpenCV - size: {size}, deviation: {deviation}', blurOpenCV)
    cv2.waitKey(0)

lr3()
```
Эта функция считывает изображение, применяет к нему пользовательскую функцию размытия по Гауссу, а затем применяет к нему функцию размытия по Гауссу OpenCV. Затем отображаются результаты. Параметры отклонения и размера корректируются для демонстрации эффекта различных настроек.

Примечания
Пользовательская функция размытия по Гауссу реализована в образовательных целях и может быть не такой эффективной и точной, как функция OpenCV. Рекомендуется использовать функцию OpenCV для производственного кода. Функция размытия по Гауссу OpenCV ( cv2.GaussianBlur) является более оптимизированной.

# Контрольные вопросы

## 1 Принцип операции размытия в OpenCV

Операция размытия в OpenCV используется для уменьшения шума в изображении путем применения фильтра Гаусса. Эта операция также известна как сглаживание или размытие изображения. Фильтр Гаусса обладает свойством отсутствия выбросов на входе ступенчатой ​​функции при минимизации времени нарастания и спада. В контексте обработки изображений он сглаживает острые края изображений, сводя к минимуму чрезмерное размытие .

## 2 Операция матричной свертки
Операция свертки — это математическая операция, которая объединяет две функции для создания третьей функции. В контексте обработки изображений свертка — это способ улучшить характеристики изображения. Операция свертки выполняется над двумя функциями: изображением и ядром. Ядро — это небольшая матрица, которая перемещается по изображению. В каждой позиции ядро ​​умножается на пиксели изображения, чтобы создать новое изображение stackoverflow.com .

## 3 Построение матрицы свертки в размытии по Гауссу
Матрица свертки для размытия по Гауссу — это матрица, представляющая функцию Гаусса. Он создается путем взятия значений кривой Гаусса и преобразования их в квадратную матрицу. Затем матрица ядра умножается на каждый пиксель изображения stackoverflow.com .

## 4 Алгоритм размытия по Гауссу
Алгоритм размытия по Гауссу — это метод размытия изображения. Он работает путем применения к изображению фильтра Гаусса. Фильтр представляет собой матрицу, представляющую функцию Гаусса. Затем матрица свертывается с изображением, в результате чего изображение становится размытым. Эффект размытия контролируется стандартным отклонением функции Гаусса, которое определяет степень размытия .

## 5 Параметры размытия по Гауссу
Параметры функции размытия по Гауссу в OpenCV:

src: исходное изображение.
ksize: Размер ядра. Это должен быть кортеж положительных нечетных чисел, например (5,5).
sigmaX: Стандартное отклонение в направлении X.
sigmaY: Стандартное отклонение в направлении Y. Если оно равно нулю, оно устанавливается равным sigmaX.
borderType: метод экстраполяции пикселей. Это должен быть один из методов интерполяции границ shimat.github.io .

## 6 Выполнение гауссовой фильтрации с использованием библиотек OpenCV
Вы можете применить операцию размытия по Гауссу в OpenCV, используя эту cv2.GaussianBlur()функцию. Вот пример:
```python
import cv2
import numpy

# Read image
src = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)

# Apply Gaussian blur on src image
dst = cv2.GaussianBlur(src,(5,5),cv2.BORDER_DEFAULT)

# Display input and output image
cv2.imshow("Gaussian Smoothing",numpy.hstack((src, dst)))
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
В этом примере изображение считывается с помощью cv2.imread(). Размытие по Гауссу применяется с помощью cv2.GaussianBlur(), с размером ядра (5,5). Затем изображение отображается с помощью cv2.imshow()
```