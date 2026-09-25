import re

with open('fedscale/cloud/aggregation/aggregator.py', 'r') as f:
    content = f.read()

target = r"""unavailable_model_impact_events: \{self\.client_manager\.model_impact_unavailable_events\}
\"\"\"
                        logging\.info\(summary_msg\)"""

replacement = """unavailable_model_impact_events: {self.client_manager.model_impact_unavailable_events}
\"\"\"
                        logging.info(summary_msg)

                        # TEMPORAL EPSILON SUMMARY
                        temporal_enabled = getattr(self.args, 'temporal_epsilon_reclustering', False)
                        if temporal_enabled and hasattr(self.client_manager, 'temporal_eval_count'):
                            import numpy as np
                            all_d = self.client_manager.all_deltas
                            all_e = self.client_manager.all_epsilons
                            avg_delta = np.mean(all_d) if len(all_d)>0 else 0
                            med_delta = np.median(all_d) if len(all_d)>0 else 0
                            avg_eps = np.mean(all_e) if len(all_e)>0 else 0
                            med_eps = np.median(all_e) if len(all_e)>0 else 0
                            
                            t_summary_msg = f\"\"\"TEMPORAL EPSILON SUMMARY

temporal_epsilon_checks: {self.client_manager.temporal_checks}
temporal_epsilon_warmups: {self.client_manager.temporal_warmups}
significant_temporal_state_changes: {self.client_manager.temporal_significant}
insignificant_temporal_state_changes: {self.client_manager.temporal_insignificant}
reclusterings: {self.client_manager.reclusterings_triggered}
skipped_reclusterings: {self.client_manager.reclusterings_skipped_due_to_low_model_impact}
reclusterings_avoided_by_temporal_epsilon: {self.client_manager.temporal_reclusterings_avoided}
average_delta: {avg_delta:.4f}
median_delta: {med_delta:.4f}
average_epsilon: {avg_eps:.4f}
median_epsilon: {med_eps:.4f}
clients_epsilon_active: {self.client_manager.temporal_active_clients}
\"\"\"
                            logging.info(t_summary_msg)"""

new_content = re.sub(target, replacement, content)

if new_content != content:
    with open('fedscale/cloud/aggregation/aggregator.py', 'w') as f:
        f.write(new_content)
    print("Patched aggregator.py with temporal summary")
else:
    print("Failed to patch aggregator.py")
