# Models Directory

This folder stores trained models as a directory containing:

- weights.npz
- config.json

Example structure:

models/
  xor_sigmoid_v1/
    weights.npz
    config.json

## Save

In code:

nn.save_dir("models/xor_sigmoid_v1")

## Load

In code:

nn2 = SimpleNeuralNetwork.load_dir("models/xor_sigmoid_v1")

## Notes

- Directory name is the model version tag.
- weights.npz stores parameters only.
- config.json stores architecture metadata required to rebuild the model.
