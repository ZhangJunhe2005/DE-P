import importlib
import unittest


class ImportBaselineTests(unittest.TestCase):
    def test_core_modules_import(self):
        targets = {
            "config": ("config.config", "cfg"),
            "DepNetwork": ("policy.dep_network", "DepNetwork"),
            "DepBackbone": ("policy.models.backbone", "DepBackbone"),
            "DynamicObstacleAttention": ("policy.models.EkfDynPercept", "DynamicObstacleAttention"),
            "PointCloudProcessor": ("policy.models.MonteCarloCutting", "PointCloudProcessor"),
            "DepTrainer": ("policy.dep_trainer", "DepTrainer"),
            "DEPDataset": ("policy.dep_dataset", "DEPDataset"),
            "SafetyLoss": ("loss.safety_loss", "SafetyLoss"),
            "SmoothnessLoss": ("loss.smoothness_loss", "SmoothnessLoss"),
            "GuidanceLoss": ("loss.guidance_loss", "GuidanceLoss"),
        }
        for label, (module_name, symbol) in targets.items():
            with self.subTest(label=label):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, symbol))

    def test_dep_loss_import(self):
        module = importlib.import_module("loss.loss_function")
        self.assertTrue(hasattr(module, "DEPLoss"))

    @unittest.expectedFailure
    def test_legacy_yopo_loss_symbol_import(self):
        module = importlib.import_module("loss.loss_function")
        self.assertTrue(hasattr(module, "YOPOLoss"), "DE-P does not export the requested YOPOLoss symbol")

    def test_ros_node_module_import_without_master(self):
        module = importlib.import_module("test_dep_ros")
        self.assertTrue(hasattr(module, "DepNet"))


if __name__ == "__main__":
    unittest.main()

