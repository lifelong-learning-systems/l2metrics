import unittest
import l2metrics

class MyUnitTest(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def testAdd(self):
        c = 1+1
        self.assertEqual(c,2)

if __name__ == "__main__":
    unittest.main()
