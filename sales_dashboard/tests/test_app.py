import pathlib
from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parents[1] / "app.py")


def test_landing_page_without_data():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and len(at.info) == 1


def test_demo_data_renders_kpis_and_filters():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 5
    labels = [m.label for m in at.metric[:5]]
    assert labels == ["Total Sales", "Total Profit", "Total Orders", "Quantity Sold", "Profit Margin"]
    assert len(at.sidebar.multiselect) >= 3
    before = at.metric[0].value
    at.sidebar.multiselect[0].set_value([at.sidebar.multiselect[0].options[0]]).run()   # filter Region
    assert not at.exception and at.metric[0].value != before
