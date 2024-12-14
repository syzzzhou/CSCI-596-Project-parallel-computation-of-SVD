import random


def generate_random_matrix_file(filename, rows, cols, min_val=0.0, max_val=100.0):
    """
    Generates a matrix file with random values within a specified range.

    :param filename: Name of the output file.
    :param rows: Number of rows in the matrix.
    :param cols: Number of columns in the matrix.
    :param min_val: Minimum value for random numbers.
    :param max_val: Maximum value for random numbers.
    """
    with open(filename, "w") as file:
        # Write the dimensions
        file.write(f"{rows}\t{cols}\n")

        # Generate and write random matrix values
        for _ in range(rows):
            row = [random.uniform(min_val, max_val) for _ in range(cols)]
            file.write("\t".join(f"{val:.6f}" for val in row) + "\n")


# Example usage
generate_random_matrix_file("matrix_input_test.txt", 6, 3, min_val=1.0, max_val=10.0)
