python scripts/run_duck.py --orcalab \
    --orca-addr localhost:50051 \
    --asset-path assets/3e2def7b0d023a52/default_project/prefabs/robot_allcollisions_usda \
    --render-fps 20 \
    --onnx policy/BEST_alpha_walking.onnx \
    --num-envs 9 \
    --realtime 
