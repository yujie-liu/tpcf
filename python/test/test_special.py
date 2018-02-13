""" Unittest for special.py """


import unittest2
import numpy as np
import lib.special as special


class TestSpecial(unittest2.TestCase):
    """ Test class for module correlation_function """
    def test_get_distance(self):
        """ Test function correlation_function.get_distance(r1, r2, theta) """
        print("\nTest get_distance")
        funct = special.distance

        # Commutative
        print("- Commutative test ")
        self.assertAlmostEqual(funct(1.3, 3., 5.), 5.09657091987)
        self.assertAlmostEqual(funct(1.3, 5., 3.), 5.09657091987)

        # Rotate by 360 deg (or 2np.pi rad)
        print("- Rotation 360 degree test")
        self.assertAlmostEqual(funct(1.3+2*np.pi, 3., 5.),
                               5.09657091987)

        # Negative radius (Rotate by 180 deg)
        print("- Rotation 180 degreee test (negative radius)")
        self.assertAlmostEqual(funct(1.3+np.pi, 3., 5.), 6.48266649294)
        self.assertAlmostEqual(funct(1.3, -3., 5.), 6.48266649294)
        self.assertAlmostEqual(funct(1.3, 3., -5.), 6.48266649294)

        # Distance to origin
        print("- Distance to origin test")
        self.assertEqual(funct(0.8, 3., 0.), 3.)
        self.assertEqual(funct(0.8, 0., 5.), 5.)

        # Same points
        print("- Same points test")
        self.assertAlmostEqual(funct(0., 3., 3.), 0.)

    def test_hist2point(self):
        """ Test function correlation_function.hist2point() """
        print("\nTest hist2point")
        funct = special.hist2point

        # General case
        print("- General case test")
        x_edges = np.array([-4.7, -4.3, -3.9, -3.5])
        y_edges = np.array([2.5, 3.5, 4.5, 5.5])
        hist = np.array([[0, -2.4, 1.2], [6.5, 2., 3.3], [0., 0., 5.2]])
        true_array = np.array([[-4.1, 3.0, 6.5],
                                  [-4.5, 4.0, -2.4],
                                  [-4.1, 4.0, 2.0],
                                  [-4.5, 5.0, 1.2],
                                  [-4.1, 5.0, 3.3],
                                  [-3.7, 5.0, 5.2]])
        np.testing.assert_almost_equal(funct(hist, x_edges, y_edges),
                                          true_array)

        # General case without excluding zeros
        print("- General case without excluding zeros test")
        true_array = np.array([[-4.5, 3.0, 0.],
                                  [-4.1, 3.0, 6.5],
                                  [-3.7, 3.0, 0.],
                                  [-4.5, 4.0, -2.4],
                                  [-4.1, 4.0, 2.0],
                                  [-3.7, 4.0, 0.],
                                  [-4.5, 5.0, 1.2],
                                  [-4.1, 5.0, 3.3],
                                  [-3.7, 5.0, 5.2]])
        np.testing.assert_almost_equal(funct(hist, x_edges, y_edges, False),
                                          true_array)

        # Array with all zeros
        print("- All zeros test")
        hist = np.zeros_like(hist)
        np.testing.assert_almost_equal(funct(hist, x_edges, y_edges),
                                          np.array([]).reshape(0, 3))

        # Array with most zeros
        print("- Most zeros test")
        hist = np.zeros_like(hist)
        hist[1][2] = 3.1
        hist[0][1] = 2.3
        true_array = np.array([[-4.5, 4.0, 2.3],
                                  [-4.1, 5.0, 3.1]])
        np.testing.assert_almost_equal(funct(hist, x_edges, y_edges),
                                          true_array)


def main():
    """ Run unittest """
    unittest2.main()


if __name__ == "__main__":
    main()
