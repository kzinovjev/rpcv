from rpcv import rpcv
import numpy as np
import numpy.testing as npt
import pytest


def get_harmonic(n, mult, amp, phase):
    return np.cos(np.linspace(0, np.pi*2, n)*mult - phase) * amp


BEADS_1D = get_harmonic(20, 1, 1, 0) + get_harmonic(20, 3, 0.5, 1)


BEADS_3D = np.array([
    get_harmonic(20, 1, 1, 0) + get_harmonic(20, 3, 0.5, 1),
    get_harmonic(20, 1, 0.5, -1) + get_harmonic(20, 2, 0.2, 0.5) + 1,
    get_harmonic(20, 1, 3, 0) + get_harmonic(20, 4, 0.1, 0) + 3,
]).T


@pytest.mark.parametrize(
    "n_beads, result",
    (
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 3)
    )
)
def test_total_harmonics(n_beads, result):
    assert rpcv.total_harmonics(n_beads) == result


@pytest.mark.parametrize(
    "x, harmonic, result",
    (
        ([0.], 0, [[0., 1.], [0., 0.]]),
        ([1., 1.], 0, [[1., 0.5, 0.5], [0., 0., 0.]]),
        ([1., 1., 1.], 1, [[0, -0.33333333, -0.33333333,  0.66666667],
                           [0, 5.77350269e-01, -5.77350269e-01,  0]]),
    )
)
def test_get_harmonic_coefs(x, harmonic, result):
    npt.assert_almost_equal(rpcv.get_harmonic_coefs(x, harmonic),
                            result)


def test_get_harmonics_coefs_gradient():
    gradient = rpcv.get_harmonics_coefs(BEADS_1D, None)[:, :, 1:]

    for bead in range(len(BEADS_1D)):

        beads_p = np.copy(BEADS_1D)
        beads_p[bead] += 1E-6
        coefs_p = rpcv.get_harmonics_coefs(beads_p, None)[:, :, 0]

        beads_m = np.copy(BEADS_1D)
        beads_m[bead] -= 1E-6
        coefs_m = rpcv.get_harmonics_coefs(beads_m, None)[:, :, 0]

        x_gradient = gradient[:, :, bead]
        num_x_gradient = (coefs_p - coefs_m) / 2E-6
        npt.assert_almost_equal(x_gradient, num_x_gradient)


@pytest.mark.parametrize(
    "coefs, harmonic, result",
    (
            ([[0, 1], [0, 0]], 0, [[0, 1], [0, 0]]),
            (rpcv.get_harmonic_coefs([0., 1., -1.], 1),
             1,
             [[1.1547005e+00, 0, 5.7735027e-01, -5.7735027e-01],
              [-2.6179939e+00, -5.7735027e-01, 2.8867513e-01, 2.8867513e-01]])
    )
)
def test_get_harmonic(coefs, harmonic, result):
    npt.assert_almost_equal(rpcv.get_harmonic(coefs, harmonic), result)


def test_get_harmonics_gradient():
    coefs = rpcv.get_harmonics_coefs(BEADS_1D, None)
    harmonics_gradient = rpcv.get_harmonics(coefs)[:, :, 1:]

    for bead in range(len(BEADS_1D)):

        beads_p = np.copy(BEADS_1D)
        beads_p[bead] += 1E-6
        coefs_p = rpcv.get_harmonics_coefs(beads_p, None)
        harmonics_p = rpcv.get_harmonics(coefs_p)[:, :, 0]

        beads_m = np.copy(BEADS_1D)
        beads_m[bead] -= 1E-6
        coefs_m = rpcv.get_harmonics_coefs(beads_m, None)
        harmonics_m = rpcv.get_harmonics(coefs_m)[:, :, 0]

        x_gradient = harmonics_gradient[:, :, bead]
        num_x_gradient = (harmonics_p - harmonics_m) / 2E-6
        npt.assert_almost_equal(x_gradient, num_x_gradient)


def test_get_delta_gradient():
    coefs = rpcv.get_all_harmonics_coefs(BEADS_3D, None)
    harmonics = rpcv.get_all_harmonics(coefs)
    delta_gradient = rpcv.get_delta(harmonics)[1]

    for bead, cv in np.ndindex(BEADS_3D.shape):

        beads_p = np.copy(BEADS_3D)
        beads_p[bead, cv] += 1E-6
        coefs_p = rpcv.get_all_harmonics_coefs(beads_p, None)
        harmonics_p = rpcv.get_all_harmonics(coefs_p)
        delta_p = rpcv.get_delta(harmonics_p)[0]

        beads_m = np.copy(BEADS_3D)
        beads_m[bead, cv] -= 1E-6
        coefs_m = rpcv.get_all_harmonics_coefs(beads_m, None)
        harmonics_m = rpcv.get_all_harmonics(coefs_m)
        delta_m = rpcv.get_delta(harmonics_m)[0]

        x_gradient = delta_gradient[:, bead, cv]
        num_x_gradient = (delta_p - delta_m) / 2E-6

        npt.assert_almost_equal(x_gradient, num_x_gradient)


@pytest.mark.parametrize(
    "n_cvs, n_harmonics, result",
    (
        (1, 3, [[1, 2]]),
        (2, 4, [[1, 2, 3], [1, 2, 3]]),
    )
)
def test_get_phi_scale_matrix(n_cvs, n_harmonics, result):
    npt.assert_almost_equal(rpcv.get_phi_scale_matrix(n_cvs, n_harmonics),
                            result)


def test_invariance_1d():
    original_rpcvs = rpcv.get_rpcvs(BEADS_1D.reshape((20, 1)))[0]
    shifted_rpcvs = rpcv.get_rpcvs(np.roll(BEADS_1D, 3).reshape((20, 1)))[0]
    inverted_rpcvs = rpcv.get_rpcvs(BEADS_1D[::-1].reshape((20, 1)))[0]

    npt.assert_almost_equal(original_rpcvs, shifted_rpcvs)
    npt.assert_almost_equal(original_rpcvs, inverted_rpcvs)


def test_invariance_3d():
    original_rpcvs = rpcv.get_rpcvs(BEADS_3D)[0]
    shifted_rpcvs = rpcv.get_rpcvs(np.roll(BEADS_3D, 3, 0))[0]
    inverted_rpcvs = rpcv.get_rpcvs(BEADS_3D[::-1])[0]

    npt.assert_almost_equal(original_rpcvs, shifted_rpcvs)
    npt.assert_almost_equal(original_rpcvs, inverted_rpcvs)


def test_rpcvs_gradient():
    rpcvs, gradient = rpcv.get_rpcvs(BEADS_3D)

    for bead, cv in np.ndindex(BEADS_3D.shape):

        beads_p = np.copy(BEADS_3D)
        beads_p[bead, cv] += 1E-6
        rpcvs_p = rpcv.get_rpcvs(beads_p)[0]

        beads_m = np.copy(BEADS_3D)
        beads_m[bead, cv] -= 1E-6
        rpcvs_m = rpcv.get_rpcvs(beads_m)[0]

        x_gradient = gradient[:, bead, cv]
        num_x_gradient = (rpcvs_p - rpcvs_m) / 2E-6

        npt.assert_almost_equal(x_gradient, num_x_gradient)
