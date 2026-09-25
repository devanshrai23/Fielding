import re

with open('fedscale/cloud/internal/client_metadata.py', 'r') as f:
    content = f.read()

target = r"""    def register_distribution\(self, distribution\):
        self\.label_distribution = distribution"""

replacement = """    def register_distribution(self, distribution):
        self.prev_label_distribution = getattr(self, 'label_distribution', None)
        self.label_distribution = distribution"""

new_content = re.sub(target, replacement, content)

if new_content != content:
    with open('fedscale/cloud/internal/client_metadata.py', 'w') as f:
        f.write(new_content)
    print("Patched ClientMetadata")
else:
    print("Failed to patch ClientMetadata")
