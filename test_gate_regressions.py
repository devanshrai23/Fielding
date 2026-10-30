import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace

source = ast.parse(Path('fedscale/cloud/client_manager.py').read_text())
manager = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ClientManager')
method = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == '_check_model_impact')
namespace = {'logging': logging}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual gate>', 'exec'), namespace)

class GateTests(unittest.TestCase):
    def manager(self, values, smoothing=False):
        m = SimpleNamespace(args=SimpleNamespace(model_impact_aware_reclustering=True,
            model_impact_smoothing=smoothing, model_impact_smoothing_window=3,
            model_impact_threshold=.05, temporal_epsilon_reclustering=False),
            data_drifted_clients={1:[1]}, client_metadata={1:SimpleNamespace(
                evaluation_accuracies=list(enumerate(values)), top1_accuracies=[.9,.1])})
        m.getDriftedClients = lambda: {c for cs in m.data_drifted_clients.values() for c in cs}
        m.getUniqueId = lambda host, client: client
        m.check = lambda: namespace['_check_model_impact'](m)
        return m

    def test_pending_drift_reconsidered_on_fresh_evaluation(self):
        m = self.manager([.8,.79])
        self.assertFalse(m.check())
        self.assertEqual(m.data_drifted_clients, {1:[1]})
        m.client_metadata[1].evaluation_accuracies.append((2,.6))
        self.assertTrue(m.check())

    def test_repeated_call_does_not_count_again(self):
        m = self.manager([.8,.79])
        self.assertFalse(m.check())
        self.assertFalse(m.check())
        self.assertEqual(m.model_impact_checks, 1)

    def test_two_windows_can_detect_plateau_after_drop(self):
        m = self.manager([.8,.8,.8,.6,.6,.6], True)
        self.assertTrue(m.check())
        self.assertEqual(m.raw_skip_decisions, 1)

    def test_training_accuracy_does_not_drive_gate(self):
        self.assertFalse(self.manager([.8,.8]).check())

    def test_no_evaluation_preserves_original_fallback(self):
        self.assertTrue(self.manager([]).check())

if __name__ == '__main__':
    unittest.main()
