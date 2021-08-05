import cupy as cp
from numpy import vectorize
import pytest
from cusignal.test.utils import array_equal
from cusignal.radartools import ca_cfar, cfar_alpha


class TestCaCfar:
    @pytest.mark.benchmark(group="CFAR")
    @pytest.mark.parametrize("length, guard_cells, reference_cells",
                             [(100, 1, 5), (11, 2, 3), (100, 10, 20)])
    class TestOneD:
        def expected(self, length, guard_cells, reference_cells):
            out = cp.zeros(length)
            N = 2 * reference_cells
            alpha = cfar_alpha(0.001, N)
            out[guard_cells + reference_cells:
                -guard_cells - reference_cells] = cp.ones(length -
                                                          2 * guard_cells -
                                                          2 * reference_cells)
            return(alpha * out)

        def test_1d_ones(self, length, guard_cells, reference_cells):
            array = cp.ones(length)
            mask, _ = ca_cfar(array, guard_cells, reference_cells)
            key = self.expected(length, guard_cells, reference_cells)
            array_equal(mask, key)

    @pytest.mark.parametrize("length, gc, rc",
                             [(1, 1, 1), (10, 2, 3), (10, 0, 5),
                              (10, 5, 0)])
    class TestFailuresOneD:
        def test_1d_failures(self, length, gc, rc):
            with pytest.raises(ValueError):
                _, _ = ca_cfar(cp.zeros(length), gc, rc)

    @pytest.mark.benchmark(group="CFAR")
    @pytest.mark.parametrize("shape, gc, rc",
                             [((10, 10), (1, 1), (2, 2)), ((10, 100), (1, 10),
                                                           (2, 20))])
    class TestTwoD:
        def expected(self, shape, gc, rc):
            out = cp.zeros(shape)
            N = 2 * rc[0] * (2 * rc[1] + 2 * gc[1] + 1)
            N += 2 * (2 * gc[0] + 1) * rc[1]
            alpha = cfar_alpha(.001, N)
            out[gc[0] + rc[0]: -gc[0] - rc[0], gc[1] + rc[1]:
                - gc[1] - rc[1]] = cp.ones((shape[0] - 2 * gc[0] - 2 * rc[0],
                                           shape[1] - 2 * gc[1] - 2 * rc[1]))
            return(alpha * out)

        def test_2d_ones(self, shape, gc, rc):
            array = cp.ones(shape)
            mask, _ = ca_cfar(array, gc, rc)
            key = self.expected(shape, gc, rc)
            array_equal(mask, key)

    @pytest.mark.parametrize("shape, gc, rc",
                             [((3, 3), (1, 2), (1, 10)),
                              ((3, 3), (1, 1), (10, 1)),
                              ((5, 5), (3, 3), (3, 3))])
    class TestFailuresTwoD:
        def test_2d_failures(self, shape, gc, rc):
            with pytest.raises(ValueError):
                _, _ = ca_cfar(cp.zeros(shape), gc, rc)

    @pytest.mark.parametrize("shape, gc, rc, points",
                             [(10, 1, 1, (6,)), (100, 10, 20, (34, 67)),
                              ((100, 200), (5, 10), (10, 20), [(31, 45),
                                                               (50, 111)])])
    class TestDetection:
        def test_point_detection(self, shape, gc, rc, points):
            '''
            Placing points too close together can yield unexpected results.
            '''
            array = cp.zeros(shape)
            for point in points:
                array[point] = 1e3
            threshold, detections = ca_cfar(array, gc, rc)
            key = array - threshold
            f = vectorize(lambda x: True if x > 0 else False)
            key = f(key.get())
            array_equal(detections, key)
