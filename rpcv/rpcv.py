import numpy as np


def get_rpcvs(beads, n_harmonics=None, with_b=True):
    """
    :param beads: ndarray(shape=(n_beads, n_cvs))
    :param n_harmonics: int
    :param with_b: bool
    :return: ndarray(shape=(n_rpcvs)), ndarray(shape=(n_rpcvs, n_beads, n_cvs))
    """
    coefs = get_all_harmonics_coefs(beads, n_harmonics)
    harmonics = get_all_harmonics(coefs)

    a, a_grad = get_a(harmonics)
    if not with_b:
        return flatten_a(a, a_grad)
    
    delta, delta_grad = get_delta(harmonics)
    b, b_grad = get_b(a, a_grad, delta, delta_grad)
    a, a_grad = flatten_a(a, a_grad)

    return np.concatenate((a, b)), np.concatenate((a_grad, b_grad))


def get_b(a, a_grad, delta, delta_grad):
    """
    :param a: ndarray(shape=(n_cvs, n_harmonics))
    :param a_grad: ndarray(shape=(n_cvs, n_harmonics, n_beads, n_cvs))
    :param delta: ndarray(shape=(n_cvs * (n_harmonics-1) - 1)
    :param delta_grad: ndarray(shape=(n_cvs * (n_harmonics-1) - 1, n_beads, n_cvs))
    :return: (ndarray(shape=(n_cvs * (n_harmonics-1) - 1),
              ndarray(shape=(n_cvs * (n_harmonics-1) - 1, n_beads, n_cvs)))
    """
    n_cvs, n_harmonics, n_beads = a_grad.shape[:3]
    grad_shape = (n_cvs * (n_harmonics-1), n_beads, n_cvs)

    a_ref = a[0, 1]
    a_ref_grad = a_grad[0, 1]

    a_b = a[:, 1:].flatten()[1:]
    a_b_grad = a_grad[:, 1:].reshape(grad_shape)[1:]

    aa = a_b * a_ref
    aa_grad = a_ref * a_b_grad + np.einsum('i,jk->ijk', a_b, a_ref_grad)

    b = np.sqrt(a_b**2 + a_ref**2 - 2 * aa * np.cos(delta))
    b_grad = np.zeros(aa_grad.shape)
    for i in range(b_grad.shape[0]):
        b_grad[i] = (a_b[i] * a_b_grad[i] +
                     a_ref * a_ref_grad +
                     aa[i] * np.sin(delta[i]) * delta_grad[i] -
                     np.cos(delta[i]) * aa_grad[i]) / b[i]

    return b, b_grad


def flatten_a(a, a_grad):
    """
    :param a: ndarray(shape=(n_cvs, n_harmonics))
    :param a_grad: ndarray(shape=(n_cvs, n_harmonics, n_beads, n_cvs))
    :return: (ndarray(shape=(n_cvs * n_harmonics)),
              ndarray(shape=(n_cvs * n_harmonics, n_beads, n_cvs)))
    """
    n_cvs, n_harmonics, n_beads = a_grad.shape[:3]
    return a.flatten(), a_grad.reshape((n_cvs * n_harmonics, n_beads, n_cvs))


def get_a(harmonics):
    """
    :param harmonics: ndarray(shape=(n_cvs, n_harmonics, 2, n_beads + 1))
    :return: (ndarray(shape=(n_cvs * n_harmonics)),
              ndarray(shape=(n_cvs * n_harmonics, n_beads, n_cvs)))
    """
    n_cvs, n_harmonics = harmonics.shape[:2]
    n_beads = harmonics.shape[3] - 1

    a = harmonics[:, :, 0, 0]

    a_grad = np.zeros((n_cvs, n_harmonics, n_beads, n_cvs))
    for cv in range(n_cvs):
        a_grad[cv, :, :, cv] = harmonics[cv, :, 0, 1:]

    return a, a_grad


def get_delta(harmonics):
    """
    :param harmonics: ndarray(shape=(n_cvs, n_harmonics, 2, n_beads + 1))
    :return: (ndarray(shape=(n_cvs * (n_harmonics-1) - 1),
              ndarray(shape=(n_cvs * (n_harmonics-1) - 1, n_beads, n_cvs)))
    """
    n_cvs, n_harmonics = harmonics.shape[:2]
    n_beads = harmonics.shape[3] - 1
    grad_shape = (n_cvs * (n_harmonics-1), n_beads, n_cvs)

    phi = harmonics[:, 1:, 1, 0]
    phi_grad = harmonics[:, 1:, 1, 1:]

    scale_matrix = get_phi_scale_matrix(n_cvs, n_harmonics)

    delta = project_angle(phi - phi[0, 0] * scale_matrix)
    delta_grad_matrix = np.zeros((n_cvs, n_harmonics-1, n_beads, n_cvs))
    for cv in range(n_cvs):
        for harmonic in range(n_harmonics - 1):
            d_0 = -phi_grad[0, 0] * scale_matrix[cv, harmonic]
            d_cv = phi_grad[cv, harmonic]
            delta_grad_matrix[cv, harmonic, :, 0] += d_0
            delta_grad_matrix[cv, harmonic, :, cv] += d_cv

    delta = delta.flatten()[1:]
    delta_grad = delta_grad_matrix.reshape(grad_shape)[1:]

    return delta, delta_grad


def get_phi_scale_matrix(n_cvs, n_harmonics):
    return np.repeat([np.arange(1, n_harmonics)], n_cvs, axis=0)


def project_angle(angle):
    return angle - np.pi * 2 * np.round(angle / (np.pi * 2))


def get_all_harmonics(coefs):
    """
    :param coefs: ndarray(shape=(n_cvs, n_harmonics, 2, n_beads + 1))
    :return: ndarray(shape=(n_cvs, n_harmonics, 2, n_beads + 1))
    """
    n_cvs = coefs.shape[0]
    return np.array([get_harmonics(coefs[cv]) for cv in range(n_cvs)])


def get_harmonics(coefs):
    """
    :param coefs: ndarray(shape=(n_harmonics, 2, n_beads + 1))
    :return: ndarray(shape=(n_harmonics, 2, n_beads + 1))
    """
    n_harmonics = coefs.shape[0]
    return np.array([get_harmonic(coefs[h], h) for h in range(n_harmonics)])


def get_harmonic(coefs, harmonic):
    """
    :param coefs: ndarray(shape=(n_harmonics, 2, n_beads + 1))
    :param harmonic: int
    :return: ndarray(shape=(2, n_beads + 1))
    """
    if harmonic == 0:
        return coefs

    result = np.zeros(coefs.shape)
    a2 = coefs[0, 0] ** 2 + coefs[1, 0] ** 2

    result[0, 0] = np.sqrt(a2)
    result[0, 1:] = (coefs[0, 0] * coefs[0, 1:] +
                     coefs[1, 0] * coefs[1, 1:]) / result[0, 0]

    result[1, 0] = np.arctan2(coefs[1, 0], coefs[0, 0])
    result[1, 1:] = (coefs[0, 0] * coefs[1, 1:] -
                     coefs[1, 0] * coefs[0, 1:]) / a2

    return result


def get_all_harmonics_coefs(beads, n_harmonics):
    """
    :param beads:  ndarray(shape=(n_beads, n_cvs))
    :param n_harmonics: int
    :return: ndarray(shape=(n_cvs, n_harmonics, 2, n_beads + 1))
    """
    return np.array([get_harmonics_coefs(x, n_harmonics) for x in beads.T])


def get_harmonics_coefs(x, n_harmonics):
    """
    :param x:  ndarray(shape=(n_beads,))
    :param n_harmonics: int
    :return: ndarray(shape=(n_harmonics, 2, n_beads + 1))
    """
    n_harmonics = n_harmonics or total_harmonics(len(x))
    return np.array([get_harmonic_coefs(x, h) for h in np.arange(n_harmonics)])


def get_harmonic_coefs(x, harmonic):
    """
    :param x: ndarray(shape=(n_beads,))
    :param harmonic: int
    :return: ndarray(shape=(2, n_beads + 1))
    """
    n_beads = len(x)
    if harmonic >= total_harmonics(n_beads):
        raise ValueError("Harmonic does not exist")

    result = np.zeros((2, n_beads + 1))

    if harmonic == 0:
        result[0, 0] = np.mean(x)
        result[0, 1:] = np.ones(n_beads) / n_beads
        return result

    j = np.arange(1, n_beads + 1)
    w_n = 2 * np.pi * harmonic / n_beads
    cos_wj = np.cos(w_n * j)
    sin_wj = np.sin(w_n * j)
    result[0, 0] = np.dot(x, cos_wj)
    result[0, 1:] = cos_wj
    result[1, 0] = np.dot(x, sin_wj)
    result[1, 1:] = sin_wj

    return result * 2 / n_beads


def total_harmonics(n_beads):
    return int(np.floor((n_beads + 1) / 2.))
