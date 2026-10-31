"""Separate Auxo entry point; FIELDING's default entry point is unchanged."""
import copy
from fedscale.cloud.aggregation.aggregator import Aggregator, parser
from fedscale.cloud.auxo.client_manager import AuxoClientManager


class AuxoAggregator(Aggregator):
    def init_client_manager(self, args):
        if args.sample_mode != 'random' or args.use_gradient_cluster:
            raise ValueError('Auxo adapter requires random sampling and use_gradient_cluster disabled')
        return AuxoClientManager(args.sample_mode, args, sample_seed=args.sample_seed)

    def client_completion_handler(self, results, cluster_id=0):
        if results.get('success', False) and cluster_id > 0:
            self.client_manager.record_update(results['client_id'], cluster_id,
                                               results['update_weight'],
                                               self.model_wrapper[cluster_id].get_weights())
        super().client_completion_handler(results, cluster_id)

    def get_cluster_avg_model(self, cluster_id):
        # Auxo child models inherit the parent; do not mix source-cohort models.
        return None


if __name__ == '__main__':
    AuxoAggregator(parser.args).run()
