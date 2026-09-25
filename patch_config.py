import re

with open('fedscale/cloud/config_parser.py', 'r') as f:
    content = f.read()

new_args = """parser.add_argument('--temporal_epsilon_reclustering', type=bool, default=False, help='Enable temporal epsilon check')
parser.add_argument('--temporal_epsilon_window', type=int, default=50, help='Rolling history window size for epsilon')
parser.add_argument('--persistence_window', type=int, default=5, help='Window for calculating drift persistence')
parser.add_argument('--persistence_threshold', type=float, default=0.8, help='Threshold for drift persistence')
parser.add_argument('--epsilon_percentile', type=int, default=95, help='Percentile for epsilon calculation')

# The basic configuration"""

content = re.sub(r'# The basic configuration', new_args, content)

with open('fedscale/cloud/config_parser.py', 'w') as f:
    f.write(content)
print("Patched config_parser")
