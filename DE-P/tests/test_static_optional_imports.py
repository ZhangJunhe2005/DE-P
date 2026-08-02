import subprocess
import sys
import textwrap
import unittest


class StaticOptionalDependencyTests(unittest.TestCase):
    def test_static_network_runs_when_dynamic_optional_packages_are_blocked(self):
        code = textwrap.dedent(
            """
            import importlib.abc, sys, torch
            class BlockDynamic(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'filterpy', 'matplotlib', 'sklearn'}:
                        raise ModuleNotFoundError('blocked optional dynamic dependency: ' + fullname)
                    return None
            sys.meta_path.insert(0, BlockDynamic())
            from policy.dep_network import DepNetwork
            model = DepNetwork().cpu().eval()
            with torch.inference_mode():
                endstate, score = model(torch.zeros(1,1,96,160), torch.zeros(1,9,3,5))
            assert endstate.shape == (1,9,3,5)
            assert score.shape == (1,3,5)
            """
        )
        completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_dynamic_backbone_reports_missing_optional_dependency(self):
        code = textwrap.dedent(
            """
            import importlib.abc, sys
            class BlockDynamic(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] == 'filterpy':
                        raise ModuleNotFoundError('blocked optional dynamic dependency: ' + fullname)
                    return None
            sys.meta_path.insert(0, BlockDynamic())
            from policy.models.backbone import AttentionDepBackbone
            try:
                AttentionDepBackbone(64)
            except ImportError as exc:
                assert 'requirements-dynamic.txt' in str(exc)
            else:
                raise AssertionError('dynamic backbone unexpectedly constructed')
            """
        )
        completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
