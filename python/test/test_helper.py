""" Unittest for helper.py """


import unittest2
import numpy as np
import configparser
from lib.cosmology import Cosmology
from lib.helper import JobHelper, CorrelationHelper, Bins

PI = 3.14159265359


class TestHelper(unittest2.TestCase):
    """ Unittest for helper class """
    def test_job_helper(self):
        """ Test JobHelper class """
        print("\nTest JobHelper class")

        # Test attributes and constructor
        # Check for exception if total job is less than 0
        print("- Attribute test")
        with self.assertRaises(ValueError) as context:
            JobHelper(-1)
        self.assertTrue('Total jobs must be at least 1.' in str(context.exception))
        helper = JobHelper(10)
        self.assertEqual(helper.current_job, 0)
        self.assertEqual(helper.total_jobs, 10)

        # Test method JobHelper.set_current_job
        print("- Set current job test ")
        with self.assertRaises(ValueError) as context:
            helper.set_current_job(-1)
        self.assertTrue('Job must be at least 0 and less than total job.' in str(context.exception))

        # Test method JobHelper.increment
        print("- Increment job test")
        helper.increment()
        self.assertEqual(helper.current_job, 1)
        helper.set_current_job(9)  # increment after reachng max
        helper.increment()
        self.assertEqual(helper.current_job, 9)

        # Test JobHelper.get_index_range
        print("- Get index range test")
        with self.assertRaises(ValueError) as context:
            helper.get_index_range(9)
        self.assertTrue('Size must be at least total jobs.' in str(context.exception))

        helper.set_current_job(0)
        true_range = (0, 123456)
        self.assertEqual(helper.get_index_range(1234569), true_range)

        helper.set_current_job(9)
        true_range = (1111112, 1234569)
        self.assertEqual(helper.get_index_range(1234569), true_range)

    def test_bins(self):
        """ Test Bins class """
        print("\nTest Bins class")

        # Setting up cosmology and configuration file
        config = configparser.SafeConfigParser()
        config.read('test/test_config.cfg')
        cosmo = Cosmology(config['COSMOLOGY'])

        bins = Bins(config['LIMIT'], config['BINWIDTH'], cosmo)

        # Test min, max
        print("- Min test")
        self.assertAlmostEqual(bins.min('ra'), np.deg2rad(110.))
        self.assertAlmostEqual(bins.min('dec'), np.deg2rad(-4))
        self.assertAlmostEqual(bins.min('theta'), 0.)
        self.assertAlmostEqual(bins.min('r'),1083.837158235826)
        self.assertAlmostEqual(bins.min('s'), 0.)

        print("- Max test")
        self.assertAlmostEqual(bins.max('ra'), np.deg2rad(260.))
        self.assertAlmostEqual(bins.max('dec'), np.deg2rad(55.))
        self.assertAlmostEqual(bins.max('theta'), np.deg2rad(10.))
        self.assertAlmostEqual(bins.max('r'), 1748.1408115843535)
        self.assertAlmostEqual(bins.max('s'), 200.)

        # Test nbins
        print("- Number of bins test")
        self.assertEqual(bins.nbins('ra'), 600)
        self.assertEqual(bins.nbins('dec'), 237)
        self.assertEqual(bins.nbins('theta'), 100)
        self.assertEqual(bins.nbins('r'), 665)
        self.assertEqual(bins.nbins('s'), 50)

        # Test binw
        print("- Binwidth test")
        self.assertAlmostEqual(bins.binw('ra'), 0.004363323129985824)
        self.assertAlmostEqual(bins.binw('dec'), 0.00434491248386774)
        self.assertAlmostEqual(bins.binw('r'), 0.9989528621782366)
        self.assertAlmostEqual(bins.binw('s'), 4.0)
        self.assertAlmostEqual(bins.binw('theta'), 0.0017453292519943294)

        # Test default binning
        print("- Default binwidth scheme test")
        self.assertEqual(bins.default_binw('s'), 2.00)
        true_angle = 0.9989528621782366/1748.1408115843535
        self.assertEqual(bins.default_binw('ra', binw_r=0.9989528621782366), true_angle)
        self.assertEqual(bins.default_binw('dec', binw_r=0.9989528621782366), true_angle)
        self.assertEqual(bins.default_binw('theta', binw_r=0.9989528621782366), true_angle)
        self.assertEqual(bins.default_binw('r', binw_s=2.00), 1.00)

        # Test bins range
        print("- Get bins test")
        np.testing.assert_almost_equal(np.linspace(np.deg2rad(110.), np.deg2rad(260.), 601),
                                       bins.bins('ra'))
        np.testing.assert_almost_equal(np.linspace(np.deg2rad(-4.), np.deg2rad(55.), 238),
                                       bins.bins('dec'))
        np.testing.assert_almost_equal(np.linspace(0, np.deg2rad(10.), 101),
                                       bins.bins('theta'))
        np.testing.assert_almost_equal(np.linspace(1083.837158235826, 1748.1408115843535, 666),
                                       bins.bins('r'))
        np.testing.assert_almost_equal(np.linspace(0, 200, 51),
                                       bins.bins('s'))

    def test_correlation_helper(self):
        pass


def main():
    """ Run unittest """
    unittest2.main()


if __name__ == "__main__":
    main()
