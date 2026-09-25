import re

with open('fedscale/cloud/client_manager.py', 'r') as f:
    content = f.read()

# Replace the end part of _check_model_impact where decision is made
target = r"""        is_significant = effective_decision

        if is_significant:
            self\.model_impact_significant_events \+= 1
            self\.reclusterings_triggered \+= 1
            decision = "RECLUSTER_SMOOTHED" if \(smoothing_enabled and insufficient_history_count == 0\) else \("RECLUSTER_RAW" if not smoothing_enabled else "RECLUSTER_PARTIAL_SMOOTHED"\)
        else:
            self\.model_impact_insignificant_events \+= 1
            self\.reclusterings_skipped_due_to_low_model_impact \+= 1
            decision = "SKIP_SMOOTHED" if \(smoothing_enabled and insufficient_history_count == 0\) else \("SKIP_RAW" if not smoothing_enabled else "SKIP_PARTIAL_SMOOTHED"\)
            self\.data_drifted_clients = \{\}

        history_ready = \(insufficient_history_count == 0\) if smoothing_enabled else "N/A"

        logging\.info\(
            f"MODEL_IMPACT_CHECK: cluster_id=\{cluster_id_str\} "
            f"drifted_clients=\{len\(drifted_clients\)\} "
            f"valid_accuracy_clients=\{len\(effective_impacts\)\} "
            f"raw_avg_model_impact=\{raw_avg:\.4f\} "
            f"smoothed_avg_model_impact=\{smoothed_avg_log\} "
            f"raw_max_model_impact=\{raw_max:\.4f\} "
            f"smoothed_max_model_impact=\{smoothed_max_log\} "
            f"threshold=\{threshold\} "
            f"smoothing_enabled=\{smoothing_enabled\} "
            f"smoothing_window=\{W if smoothing_enabled else 'N/A'\} "
            f"history_ready=\{history_ready\} "
            f"final_recluster_decision=\{decision\}"
        \)

        return is_significant"""

replacement = """        is_significant = effective_decision
        
        temporal_enabled = getattr(self.args, 'temporal_epsilon_reclustering', False)
        if not hasattr(self, 'temporal_eval_count'):
            self.temporal_eval_count = 0
            self.temporal_checks = 0
            self.temporal_warmups = 0
            self.temporal_significant = 0
            self.temporal_insignificant = 0
            self.temporal_active_clients = 0
            self.temporal_reclusterings_avoided = 0
            self.all_deltas = []
            self.all_epsilons = []
        self.temporal_eval_count += 1
        
        global_temporal_significant = True
        
        if temporal_enabled and len(drifted_clients) > 0:
            temporal_sigs = []
            for client_id in drifted_clients:
                metadata = self.client_metadata[self.getUniqueId(0, client_id)]
                if not hasattr(metadata, 'delta_history'):
                    metadata.delta_history = []
                    metadata.last_M = 0.0
                    metadata.last_s = None
                    metadata.drift_eval_history = []
                    metadata.epsilon_active = False

                metadata.drift_eval_history.append(self.temporal_eval_count)
                PW = getattr(self.args, 'persistence_window', 5)
                metadata.drift_eval_history = [e for e in metadata.drift_eval_history if e > self.temporal_eval_count - PW]
                P_t = len(metadata.drift_eval_history) / float(PW)

                D_L = 0.0
                if hasattr(metadata, 'prev_label_distribution') and metadata.prev_label_distribution is not None:
                    try:
                        from scipy.spatial.distance import jensenshannon
                        import numpy as np
                        p = np.array(metadata.prev_label_distribution)
                        q = np.array(metadata.label_distribution)
                        if sum(p) > 0: p = p / sum(p)
                        if sum(q) > 0: q = q / sum(q)
                        D_L = jensenshannon(p, q)
                        if np.isnan(D_L): D_L = 0.0
                    except:
                        pass

                D_X = 0.0
                D_C = 0.0
                M_t = D_L + D_X + D_C
                V_t = M_t - metadata.last_M
                
                norm_s_t = [D_L, D_X, D_C, M_t, (V_t + 1.0) / 2.0, P_t]
                delta_t = 0.0
                if metadata.last_s is not None:
                    delta_t = sum(abs(a - b) for a, b in zip(norm_s_t, metadata.last_s))
                    
                metadata.last_s = norm_s_t
                metadata.last_M = M_t
                
                EW = getattr(self.args, 'temporal_epsilon_window', 50)
                EP = getattr(self.args, 'epsilon_percentile', 95)
                history_size = len(metadata.delta_history)
                
                client_temporal_significant = True
                epsilon_t = "N/A"
                if history_size >= EW:
                    if not metadata.epsilon_active:
                        metadata.epsilon_active = True
                        self.temporal_active_clients += 1
                    
                    import numpy as np
                    epsilon_t = np.percentile(metadata.delta_history[-EW:], EP)
                    client_temporal_significant = bool(delta_t > epsilon_t)
                    
                    self.temporal_checks += 1
                    if client_temporal_significant:
                        self.temporal_significant += 1
                    else:
                        self.temporal_insignificant += 1
                        
                    self.all_deltas.append(delta_t)
                    self.all_epsilons.append(epsilon_t)
                    
                    logging.info(f"TEMPORAL_EPSILON_CHECK: client_id={client_id} round={self.temporal_eval_count} DL={D_L:.4f} DX={D_X:.4f} DC={D_C:.4f} M={M_t:.4f} V={V_t:.4f} P={P_t:.4f} delta={delta_t:.4f} epsilon={epsilon_t:.4f} history_size={history_size} temporal_state_change_significant={client_temporal_significant}")
                else:
                    self.temporal_warmups += 1
                    logging.info(f"TEMPORAL_EPSILON_WARMUP: client_id={client_id} round={self.temporal_eval_count} history_size={history_size} required_history={EW} delta={delta_t:.4f} decision=USE_EXISTING_MODEL_IMPACT_LOGIC")
                
                metadata.delta_history.append(delta_t)
                if len(metadata.delta_history) > EW:
                    metadata.delta_history.pop(0)
                    
                temporal_sigs.append(client_temporal_significant)
                
            global_temporal_significant = any(temporal_sigs)

        final_recluster = is_significant
        if temporal_enabled:
            final_recluster = is_significant and global_temporal_significant

        if final_recluster:
            self.model_impact_significant_events += 1
            self.reclusterings_triggered += 1
            decision = "RECLUSTER_SMOOTHED" if (smoothing_enabled and insufficient_history_count == 0) else ("RECLUSTER_RAW" if not smoothing_enabled else "RECLUSTER_PARTIAL_SMOOTHED")
            if temporal_enabled:
                decision += "_AND_TEMPORAL_SIG"
        else:
            self.model_impact_insignificant_events += 1
            self.reclusterings_skipped_due_to_low_model_impact += 1
            
            if temporal_enabled and is_significant and not global_temporal_significant:
                decision = "SKIP_TEMPORAL_EPSILON"
                self.temporal_reclusterings_avoided += 1
            else:
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

        return final_recluster"""

new_content = re.sub(target, replacement, content)

if new_content != content:
    with open('fedscale/cloud/client_manager.py', 'w') as f:
        f.write(new_content)
    print("Patched _check_model_impact with temporal epsilon")
else:
    print("Failed to patch _check_model_impact")
