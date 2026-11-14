def pytest_configure(config):
    config.addinivalue_line("markers", "reference: runs the reference policy against real scenes (slower)")
