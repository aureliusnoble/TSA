from experiments import common


def test_val_pages_inventory():
    pages = common.val_pages()
    assert len(pages) >= 20
    for p in pages[:3]:
        assert p["rows_gt"].exists() and p["cols_gt"].exists() and p["source"].exists()


def test_load_gt_bands_scales_to_work():
    pages = common.val_pages()
    # pages[0] (Aisne_Fère-en-Tardenois_1810-1825_64) has a single annotated
    # line pair (2 bands); use pages[1], a fully annotated table, instead.
    bands = common.load_gt_bands(pages[1]["rows_gt"], axis="y", work_w=3840, work_h=2880)
    assert len(bands) >= 5
    assert all(0 <= s < e <= 2880 * 1.02 for s, e in bands)
