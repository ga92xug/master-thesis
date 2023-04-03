import numpy as np
from scipy.ndimage import affine_transform


def linear_transform_array_nd(x, trafo: np.ndarray, exact=True, order=2):
    n = trafo.shape[0]
    assert trafo.shape == (n, n)
    assert len(x.shape) >= n

    # TODO : MAKE THIS EXPLICIT SOMEWHERE IN THE DOCS!!!
    # assume trafo matrix has [X, Y, Z, ....] order
    # but input tensor has [..., -Z, -Y, X] order
    trafo = trafo[::-1, ::-1].copy()
    trafo[:-1, :] *= -1
    trafo[:, :-1] *= -1

    D = len(x.shape)
    at = np.abs(trafo)

    if exact and (
        np.isclose(at.sum(axis=0), 1).all()
        and np.isclose(at.sum(axis=1), 1).all()
        and (np.isclose(at, 1.0) | np.isclose(at, 0.0)).all()
    ):
        # if it is a permutation matrix we can perform this transformation without interpolation
        axs = np.around(trafo).astype(np.int) @ np.arange(1, n + 1).reshape(n, 1)
        axs = axs.reshape(-1)

        stride = np.sign(axs).tolist()
        axs = np.abs(axs).tolist()

        axs = list(range(D - n)) + [D - n - 1 + a for a in axs]
        assert len(axs) == D, (len(axs), D)

        y = x.transpose(axs)

        stride = (Ellipsis,) + tuple([slice(None, None, s) for s in stride])
        y = y[stride]
        return y
    else:
        trafo = trafo.T

        t = np.eye(D)
        t[-n:, -n:] = trafo
        center = np.zeros(len(x.shape))
        center[-n:] = (np.asarray(x.shape[-n:]) - 1) / 2
        center[-n:] = -(trafo - np.eye(n)) @ center[-n:]

        return affine_transform(x, t, offset=center, order=order)

# Florian Slack 3.4.2023 
def closest_divisor(number, div):
    """
    Get next closest divisor to given number and divisor. Take second closest
    divisor if initial one results in number smaller than 3.
    We set the value 3 heuristically as group normalization with groups of less
    than 4 channels seem to perform worse.
    """
    all_divs = [i for i in range(1, number + 1) if number % i == 0]
    if len(all_divs) < 2:
        closest_div = all_divs[0]
    else:
        idx = np.argsort(np.abs(np.array(all_divs) - div))[:2]
        # Check next closest divisor if result is too small
        if int(number / all_divs[idx[0]]) > 3:
            return all_divs[idx[0]]
        closest_div = all_divs[idx[1]]
    # Set divisor to 1 if result is too small
    if int(number / closest_div) <= 3:
        return 1
    return closest_div