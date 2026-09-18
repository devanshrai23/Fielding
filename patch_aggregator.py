import re

with open('fedscale/cloud/aggregation/aggregator.py', 'r') as f:
    content = f.read()

target = r"""                    if hasattr\(self\.client_manager, 'model_impact_checks'\):
                        logging\.info\(f"MODEL_IMPACT_SUMMARY\\n        data_drift_events=\{self\.client_manager\.total_data_drift_events\}\\n        model_impact_checks=\{self\.client_manager\.model_impact_checks\}\\n        significant_model_impact=\{self\.client_manager\.model_impact_significant_events\}\\n        insignificant_model_impact=\{self\.client_manager\.model_impact_insignificant_events\}\\n        unavailable_model_impact=\{self\.client_manager\.model_impact_unavailable_events\}\\n        reclusterings=\{self\.client_manager\.reclusterings_triggered\}\\n        skipped_reclusterings=\{self\.client_manager\.reclusterings_skipped_due_to_low_model_impact\}"\)"""

new_summary = """                    if hasattr(self.client_manager, 'model_impact_checks'):
                        smoothing_enabled = getattr(self.args, 'model_impact_smoothing', False)
                        W = getattr(self.args, 'model_impact_smoothing_window', 3)
                        
                        summary_msg = f\"\"\"MODEL IMPACT SMOOTHING SUMMARY

smoothing_enabled: {smoothing_enabled}
window: {W if smoothing_enabled else 'N/A'}

total_model_impact_checks: {self.client_manager.model_impact_checks}
total_data_drift_events: {self.client_manager.total_data_drift_events}

raw_recluster_decisions: {self.client_manager.raw_recluster_decisions}
smoothed_recluster_decisions: {self.client_manager.smoothed_recluster_decisions if smoothing_enabled else 'N/A'}

raw_skip_decisions: {self.client_manager.raw_skip_decisions}
smoothed_skip_decisions: {self.client_manager.smoothed_skip_decisions if smoothing_enabled else 'N/A'}

decisions_changed_by_smoothing: {self.client_manager.decisions_changed_by_smoothing if smoothing_enabled else 'N/A'}
reclusterings_avoided_by_smoothing: {self.client_manager.reclusterings_avoided_by_smoothing if smoothing_enabled else 'N/A'}

insufficient_history_events: {self.client_manager.insufficient_history_events if smoothing_enabled else 'N/A'}
unavailable_model_impact_events: {self.client_manager.model_impact_unavailable_events}
\"\"\"
                        logging.info(summary_msg)"""

new_content = re.sub(target, new_summary, content)

if new_content != content:
    with open('fedscale/cloud/aggregation/aggregator.py', 'w') as f:
        f.write(new_content)
    print("Successfully patched aggregator.py")
else:
    print("Could not find the target block in aggregator.py")
