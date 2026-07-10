from experiments import metrics


def test_match_prf_exact():
    boxes = [(0, 0, 10, 10), (20, 0, 10, 10)]
    r = metrics.match_prf(boxes, boxes, iou_t=0.5)
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0
    assert r["f1"] == 1.0 and r["mean_iou"] == 1.0


def test_match_prf_partial():
    pred = [(0, 0, 10, 10), (100, 100, 10, 10)]
    gt = [(1, 0, 10, 10), (50, 50, 10, 10)]
    r = metrics.match_prf(pred, gt, iou_t=0.5)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_cell_report_counts():
    gt = {"col1_row1": (0, 0, 10, 10), "col2_row1": (10, 0, 10, 10)}
    pred = dict(gt)
    rep = metrics.cell_report(pred, gt)
    assert rep["f1_50"] == 1.0 and rep["count_err"] == 0
    assert rep["row_count_err"] == 0 and rep["col_count_err"] == 0
