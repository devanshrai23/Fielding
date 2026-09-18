import re

with open('fedscale/cloud/client_manager.py', 'r') as f:
    content = f.read()

# Define the new method
new_method = """    def _check_model_impact(self):
        if getattr(self.args, 'model_impact_aware_reclustering', False) == False:
            return True

        if not hasattr(self, 'model_impact_checks'):
            self.model_impact_checks = 0
            
            self.raw_recluster_decisions = 0
            self.smoothed_recluster_decisions = 0
            self.raw_skip_decisions = 0
            self.smoothed_skip_decisions = 0
            
            self.decisions_changed_by_smoothing = 0
            self.reclusterings_avoided_by_smoothing = 0
            self.model_impact_unavailable_events = 0
            self.insufficient_history_events = 0
            
            self.total_data_drift_events = 0

            # Keep for backward compatibility with aggregator summary if needed
            self.model_impact_significant_events = 0
            self.model_impact_insignificant_events = 0
            self.reclusterings_triggered = 0
            self.reclusterings_skipped_due_to_low_model_impact = 0

        self.total_data_drift_events += 1
        self.model_impact_checks += 1

        smoothing_enabled = getattr(self.args, 'model_impact_smoothing', False)
        W = getattr(self.args, 'model_impact_smoothing_window', 3)
        if smoothing_enabled and W < 2:
            raise ValueError("model_impact_smoothing_window must be >= 2")

        drifted_clients = self.getDriftedClients()
        raw_impacts = []
        smoothed_impacts = []
        effective_impacts = []
        
        insufficient_history_count = 0

        for client_id in drifted_clients:
            metadata = self.client_metadata[self.getUniqueId(0, client_id)]
            accuracies = metadata.top1_accuracies
            
            # 1. Compute raw impact (if possible)
            client_raw = None
            if len(accuracies) >= 2:
                prev_acc = accuracies[-2]
                curr_acc = accuracies[-1]
                client_raw = max(0, prev_acc - curr_acc)
                raw_impacts.append(client_raw)

            # 2. Compute smoothed impact (if enabled and possible)
            client_smoothed = None
            if smoothing_enabled:
                if len(accuracies) >= 2 * W:
                    recent_window = accuracies[-W:]
                    previous_window = accuracies[-2*W:-W]
                    recent_avg = sum(recent_window) / float(W)
                    previous_avg = sum(previous_window) / float(W)
                    client_smoothed = max(0, previous_avg - recent_avg)
                    smoothed_impacts.append(client_smoothed)
                else:
                    insufficient_history_count += 1
                    
            # 3. Determine effective impact for this client
            if smoothing_enabled:
                if client_smoothed is not None:
                    effective_impacts.append(client_smoothed)
                elif client_raw is not None:
                    effective_impacts.append(client_raw)
            else:
                if client_raw is not None:
                    effective_impacts.append(client_raw)

        cluster_id_str = ",".join(map(str, self.data_drifted_clients.keys()))
        threshold = getattr(self.args, 'model_impact_threshold', 0.05)

        # Fallback if NO valid metrics are available
        if len(effective_impacts) == 0:
            self.model_impact_unavailable_events += 1
            self.reclusterings_triggered += 1
            self.raw_recluster_decisions += 1
            self.model_impact_significant_events += 1
            if smoothing_enabled:
                self.smoothed_recluster_decisions += 1
                
            logging.info(f"MODEL_IMPACT_CHECK: cluster_id={cluster_id_str} drifted_clients={len(drifted_clients)} valid_accuracy_clients=0 raw_avg_model_impact=N/A smoothed_avg_model_impact=N/A raw_max_model_impact=N/A smoothed_max_model_impact=N/A threshold={threshold} smoothing_enabled={smoothing_enabled} smoothing_window={W if smoothing_enabled else 'N/A'} history_ready=False final_recluster_decision=FALLBACK_ORIGINAL_NO_MODEL_IMPACT_DATA")
            return True

        if smoothing_enabled and insufficient_history_count > 0:
            self.insufficient_history_events += 1

        # Calculate averages & maxes
        raw_avg = (sum(raw_impacts) / len(raw_impacts)) if len(raw_impacts) > 0 else 0
        raw_max = max(raw_impacts) if len(raw_impacts) > 0 else 0
        
        smoothed_avg_log = "N/A"
        smoothed_max_log = "N/A"
        if smoothing_enabled and len(smoothed_impacts) > 0:
            smoothed_avg_log = f"{(sum(smoothed_impacts) / len(smoothed_impacts)):.4f}"
            smoothed_max_log = f"{max(smoothed_impacts):.4f}"

        effective_avg = sum(effective_impacts) / len(effective_impacts)
        
        # Raw decision (counterfactual)
        raw_decision = (raw_avg >= threshold) or (len(raw_impacts) == 0)
        
        # Effective decision (actual)
        effective_decision = effective_avg >= threshold

        # Update counters
        if raw_decision:
            self.raw_recluster_decisions += 1
        else:
            self.raw_skip_decisions += 1
            
        if smoothing_enabled:
            if effective_decision:
                self.smoothed_recluster_decisions += 1
            else:
                self.smoothed_skip_decisions += 1
                
            if raw_decision and not effective_decision:
                self.decisions_changed_by_smoothing += 1
                self.reclusterings_avoided_by_smoothing += 1
            elif not raw_decision and effective_decision:
                self.decisions_changed_by_smoothing += 1

        is_significant = effective_decision

        if is_significant:
            self.model_impact_significant_events += 1
            self.reclusterings_triggered += 1
            decision = "RECLUSTER_SMOOTHED" if (smoothing_enabled and insufficient_history_count == 0) else ("RECLUSTER_RAW" if not smoothing_enabled else "RECLUSTER_PARTIAL_SMOOTHED")
        else:
            self.model_impact_insignificant_events += 1
            self.reclusterings_skipped_due_to_low_model_impact += 1
            decision = "SKIP_SMOOTHED" if (smoothing_enabled and insufficient_history_count == 0) else ("SKIP_RAW" if not smoothing_enabled else "SKIP_PARTIAL_SMOOTHED")
            self.data_drifted_clients = {}

        history_ready = (insufficient_history_count == 0) if smoothing_enabled else "N/A"

        logging.info(
            f"MODEL_IMPACT_CHECK: cluster_id={cluster_id_str} "
            f"drifted_clients={len(drifted_clients)} "
            f"valid_accuracy_clients={len(effective_impacts)} "
            f"raw_avg_model_impact={raw_avg:.4f} "
            f"smoothed_avg_model_impact={smoothed_avg_log} "
            f"raw_max_model_impact={raw_max:.4f} "
            f"smoothed_max_model_impact={smoothed_max_log} "
            f"threshold={threshold} "
            f"smoothing_enabled={smoothing_enabled} "
            f"smoothing_window={W if smoothing_enabled else 'N/A'} "
            f"history_ready={history_ready} "
            f"final_recluster_decision={decision}"
        )

        return is_significant
"""

# Extract everything before def _check_model_impact and after it
match = re.search(r'(?s)(.*?)(    def _check_model_impact\(self\):.*?)(    def clientReclusterAllGradientBased.*)', content)

if match:
    new_content = match.group(1) + new_method + "\n" + match.group(3)
    with open('fedscale/cloud/client_manager.py', 'w') as f:
        f.write(new_content)
    print("Successfully patched client_manager.py")
else:
    print("Could not find the target block in client_manager.py")
