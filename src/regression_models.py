def solve_gauss_jordan(A, b):
    print("[+] Solving Normal Equation with general matrix method...")

    n = len(A)
    matrix = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot_row = max(
            range(col, n),
            key=lambda row: abs(matrix[row][col])
        )

        if abs(matrix[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix.")

        matrix[col], matrix[pivot_row] = (
            matrix[pivot_row],
            matrix[col]
        )

        pivot = matrix[col][col]
        matrix[col] = [
            value / pivot
            for value in matrix[col]
        ]

        for row in range(n):
            if row != col:
                factor = matrix[row][col]
                matrix[row] = [
                    current - factor * pivot_value
                    for current, pivot_value
                    in zip(matrix[row], matrix[col])
                ]

    return [matrix[i][n] for i in range(n)]


def fit_normal_equation(X, y):
    print("[+] Fitting two-feature Normal Equation model...")

    X_with_intercept = [[1.0] + row for row in X]

    n_features = len(X_with_intercept[0])

    xtx = [
        [
            sum(row[i] * row[j] for row in X_with_intercept)
            for j in range(n_features)
        ]
        for i in range(n_features)
    ]

    xty = [
        sum(row[i] * target for row, target in zip(X_with_intercept, y))
        for i in range(n_features)
    ]

    weights = solve_gauss_jordan(xtx, xty)

    print(
        f"[+] Normal Equation fitted: "
        f"w0={weights[0]:.2f}, "
        f"w1={weights[1]:.2f}, "
        f"w2={weights[2]:.2f}"
    )

    return weights


def fit_normal_equation_single_feature(x, y):
    print("[+] Fitting BMI-only baseline...")

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(v * v for v in x)
    sum_xy = sum(vx * vy for vx, vy in zip(x, y))

    a = float(n)
    b = sum_x
    c = sum_x
    d = sum_x2

    det = a * d - b * c

    if det == 0:
        raise ValueError("Singular matrix.")

    inv_xtx = [
        [d / det, -b / det],
        [-c / det, a / det]
    ]

    xty = [sum_y, sum_xy]

    w0 = (
        inv_xtx[0][0] * xty[0]
        + inv_xtx[0][1] * xty[1]
    )

    w1 = (
        inv_xtx[1][0] * xty[0]
        + inv_xtx[1][1] * xty[1]
    )

    print(
        f"[+] BMI-only baseline fitted: "
        f"w0={w0:.2f}, w1={w1:.2f}"
    )

    return w0, w1


def standardize(values):
    mean = sum(values) / len(values)

    std = (
        sum((value - mean) ** 2 for value in values)
        / len(values)
    ) ** 0.5

    if std == 0:
        raise ValueError("Cannot standardize a constant feature.")

    standardized = [
        (value - mean) / std
        for value in values
    ]

    return standardized, mean, std


def fit_gradient_descent(X, y, learning_rate=0.05, epochs=10000):
    print("[+] Standardizing BMI and age...")

    bmi = [row[0] for row in X]
    age = [row[1] for row in X]

    bmi_std, bmi_mean, bmi_scale = standardize(bmi)
    age_std, age_mean, age_scale = standardize(age)

    X_std = [
        [b, a]
        for b, a in zip(bmi_std, age_std)
    ]

    print("[+] Training Gradient Descent model...")

    w0 = 0.0
    w1 = 0.0
    w2 = 0.0

    n = len(y)

    for _ in range(epochs):
        predictions = [
            w0 + w1 * row[0] + w2 * row[1]
            for row in X_std
        ]

        error = [
            prediction - target
            for prediction, target in zip(predictions, y)
        ]

        dw0 = sum(error) / n

        dw1 = sum(
            e * row[0]
            for e, row in zip(error, X_std)
        ) / n

        dw2 = sum(
            e * row[1]
            for e, row in zip(error, X_std)
        ) / n

        w0 -= learning_rate * dw0
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2

    w1_original = w1 / bmi_scale
    w2_original = w2 / age_scale

    w0_original = (
        w0
        - (w1 * bmi_mean / bmi_scale)
        - (w2 * age_mean / age_scale)
    )

    print(
        f"[+] Gradient Descent fitted: "
        f"w0={w0_original:.2f}, "
        f"w1={w1_original:.2f}, "
        f"w2={w2_original:.2f}"
    )

    return w0_original, w1_original, w2_original


def predict(X, weights):
    w0, w1, w2 = weights

    return [
        w0 + w1 * row[0] + w2 * row[1]
        for row in X
    ]


def predict_baseline(x, weights):
    w0, w1 = weights

    return [
        w0 + w1 * value
        for value in x
    ]