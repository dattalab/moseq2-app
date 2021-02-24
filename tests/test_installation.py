from unittest import TestCase

class TestInstallation(TestCase):

    def test_check_moseq2_extract(self):
        import moseq2_extract

        assert moseq2_extract.__version__ == '0.7.0'

    def test_check_moseq2_pca(self):
        import moseq2_pca

        assert moseq2_pca.__version__ == '0.4.0'

    def test_check_moseq2_model(self):
        import moseq2_model

        assert moseq2_model.__version__ == '0.5.0'


    def test_check_moseq2_viz(self):
        import moseq2_viz

        assert moseq2_viz.__version__ == '0.5.0'
