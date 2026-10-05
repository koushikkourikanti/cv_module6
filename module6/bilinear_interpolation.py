"""
CSc 8830 Assignment 6
Bilinear Interpolation

Formula:

I(x,y) =

(1-alpha)(1-beta) Q11

+ alpha(1-beta) Q21

+ (1-alpha)beta Q12

+ alpha beta Q22
"""

import numpy as np


def bilinear_sample(
    gray_image,
    x,
    y,
):

    height, width = (
        gray_image.shape[:2]
    )


    # Keep coordinates inside image
    x = float(
        np.clip(
            x,
            0,
            width - 1.000001,
        )
    )


    y = float(
        np.clip(
            y,
            0,
            height - 1.000001,
        )
    )


    x0 = int(
        np.floor(x)
    )

    y0 = int(
        np.floor(y)
    )


    x1 = min(
        x0 + 1,
        width - 1
    )

    y1 = min(
        y0 + 1,
        height - 1
    )


    alpha = x - x0

    beta = y - y0


    Q11 = float(
        gray_image[y0, x0]
    )

    Q21 = float(
        gray_image[y0, x1]
    )

    Q12 = float(
        gray_image[y1, x0]
    )

    Q22 = float(
        gray_image[y1, x1]
    )


    interpolated_value = (

        (1 - alpha)
        *
        (1 - beta)
        *
        Q11

        +

        alpha
        *
        (1 - beta)
        *
        Q21

        +

        (1 - alpha)
        *
        beta
        *
        Q12

        +

        alpha
        *
        beta
        *
        Q22
    )


    return interpolated_value


def numerical_example():

    Q11 = 100

    Q21 = 120

    Q12 = 140

    Q22 = 160


    alpha = 0.25

    beta = 0.60


    result = (

        (1 - alpha)
        *
        (1 - beta)
        *
        Q11

        +

        alpha
        *
        (1 - beta)
        *
        Q21

        +

        (1 - alpha)
        *
        beta
        *
        Q12

        +

        alpha
        *
        beta
        *
        Q22
    )


    print(
        "Bilinear Interpolation Example"
    )

    print(
        "Q11 =", Q11
    )

    print(
        "Q21 =", Q21
    )

    print(
        "Q12 =", Q12
    )

    print(
        "Q22 =", Q22
    )

    print(
        "alpha =", alpha
    )

    print(
        "beta =", beta
    )

    print(
        "Interpolated value =",
        result
    )


if __name__ == "__main__":

    numerical_example()