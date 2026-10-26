combination = [10, 20, 30, 40, 50]

# Create a linear combination of the elements in the list
linear_combination = sum(combination)
print(linear_combination)

# Create a linear combination with coefficients
coefficients = [1, 2, 3, 4, 5]
linear_combination_with_coefficients = sum(
    c * x for c, x in zip(coefficients, combination)
)
print(linear_combination_with_coefficients)

# create a linear combination with negative coefficients
negative_coefficients = [-1, -2, -3, -4, -5]
linear_combination_with_negative_coefficients = sum(
    c * x for c, x in zip(negative_coefficients, combination)
)
print(linear_combination_with_negative_coefficients)

# Create a linear combination with mixed coefficients
mixed_coefficients = [1, -2, 3, -4, 5]
linear_combination_with_mixed_coefficients = sum(
    c * x for c, x in zip(mixed_coefficients, combination)
)
print(linear_combination_with_mixed_coefficients)

# Create a linear combination with fractional coefficients
fractional_coefficients = [0.5, 1.5, 2.5, 3.5, 4.5]
linear_combination_with_fractional_coefficients = sum(
    c * x for c, x in zip(fractional_coefficients, combination)
)
print(linear_combination_with_fractional_coefficients)

# create a linear combination with zero coefficients
zero_coefficients = [0, 0, 0, 0, 0]
linear_combination_with_zero_coefficients = sum(
    c * x for c, x in zip(zero_coefficients, combination)
)
print(linear_combination_with_zero_coefficients)
