"""
Worker module for distributed FedLoRA deployment.

The worker is a thin wrapper that:
1. Reads static manifest-provided arguments
2. Loads partition artifact
3. Constructs injected ZMQClientCommManager
4. Instantiates existing FedLoRA Client
5. Calls join_in() then run()
"""
