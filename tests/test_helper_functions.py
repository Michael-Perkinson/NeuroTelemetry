from main import calculate_dynamic_bins
    

def test_calculate_dynamic_bins():
    assert calculate_dynamic_bins(10) <= 10
    assert calculate_dynamic_bins(0) == 0

    bins = calculate_dynamic_bins(5000)
    assert isinstance(bins, int)
    assert 0 < bins <= 5000

    custom = calculate_dynamic_bins(n=1000, b_ref=100, t_ref=500, k=2)
    assert isinstance(custom, int)
