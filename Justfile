server:
    RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --include-dashboard 1 --head

client:
    RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --address $SERVER_IP:6379 

kill-ray:
    uv run ray stop --force
