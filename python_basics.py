"""Задачи на базовый Python."""
from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    return sum(ch in "aeiou" for ch in data.value.lower())


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number:
        number &= number - 1
        count += 1
    return count


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        while number:
            product *= number % 10
            number //= 10
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Vectors must have the same length")
    if not predicted:
        raise ValueError("Vectors must not be empty")
    return sum((a - b) ** 2 for a, b in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number == 1:
        return "(1)"
    parts = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power:
            parts.append(f"({divisor}**{power})" if power > 1 else f"({divisor})")
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        parts.append(f"({number})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    level = 0
    while total < cube_count:
        level += 1
        total += level * level
    return level if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)
    middle = len(digits) // 2
    if len(digits) % 2:
        left, right = digits[:middle], digits[middle + 1:]
    else:
        left, right = digits[:middle - 1], digits[middle + 1:]
    return sum(map(int, left)) == sum(map(int, right))
