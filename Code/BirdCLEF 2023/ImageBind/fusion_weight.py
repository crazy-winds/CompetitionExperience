import glob
import torch

pts = glob.glob("*.pt")
models = [torch.load(pt, "cpu")["model"]  for pt in pts]
keys = list(models[0])

for key in keys:
    for i in range(1, len(models)):
        models[0][key] = models[0][key] + models[i][key]
    models[0][key] = models[0][key] / len(models)

torch.save({"model": models[0]}, "ConvNextS_fusion.pt")

