"""Задачи на NumPy."""
import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = np.asarray(data.matrices), np.asarray(data.vectors)
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    return (np.asarray(data.matrix) > data.threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    return [np.unique(row).tolist() for row in np.asarray(data.matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(matrix[:, j]).tolist() for j in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(loc=data.mean, scale=data.std, size=(data.rows, data.columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns = np.indices((data.rows, data.columns))
    return np.where((rows + columns) % 2 == 0, data.first, data.second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:] = data.background_color
    top = (data.image_height - data.height) // 2
    left = (data.image_width - data.width) // 2
    bottom = top + data.height
    right = left + data.width
    image[max(0, top):min(data.image_height, bottom), max(0, left):min(data.image_width, right)] = data.shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:] = data.background_color
    y, x = np.ogrid[:data.image_height, :data.image_width]
    x0, y0 = (data.image_width - 1) / 2, (data.image_height - 1) / 2
    mask = ((x - x0) / data.semi_axis_x) ** 2 + ((y - y0) / data.semi_axis_y) ** 2 <= 1
    image[mask] = data.shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values, dtype=float)
    maxima = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1
    minima = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])) + 1
    moving = np.convolve(values, np.ones(data.window) / data.window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    if data.class_count is None:
        classes = int(labels.max()) + 1 if labels.size else 0
    else:
        classes = data.class_count
    result = np.zeros((labels.size, classes), dtype=int)
    if labels.size:
        result[np.arange(labels.size), labels] = 1
    return result
