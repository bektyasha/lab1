"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(matrix[:, column]).tolist() for column in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    pattern = (np.indices((rows, columns)).sum(axis=0) % 2 == 0)
    return np.where(pattern, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    center_y, center_x = image_height // 2, image_width // 2
    y0 = (image_height - height) // 2
    x0 = (image_width - width) // 2
    y1 = y0 + height
    x1 = x0 + width
    y_start, y_end = max(0, y0), min(image_height, y1)
    x_start, x_end = max(0, x0), min(image_width, x1)
    if y_start < y_end and x_start < x_end:
        image[y_start:y_end, x_start:x_end] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    y, x = np.ogrid[:image_height, :image_width]
    center_y = image_height // 2
    center_x = image_width // 2
    mask = (
        ((x - center_x) ** 2) / (semi_axis_x ** 2)
        + ((y - center_y) ** 2) / (semi_axis_y ** 2)
        <= 1
    )
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values)
    local_maxima = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    ) + 1
    local_minima = np.flatnonzero(
        (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    ) + 1
    moving_average = np.convolve(
        values, np.ones(window) / window, mode="valid"
    )
    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
