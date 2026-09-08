# tests/test_sample.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_dummy():
    """اختبار وهمي للتأكد من أن pytest يعمل"""
    assert 1 + 1 == 2