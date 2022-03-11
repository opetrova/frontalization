# huggingface-integration

The [Hugging Face Hub](https://huggingface.co/docs/hub/main) hosts Git-based repositories for machine learning models, datasets, and spaces (ML demo apps).

This package contains a Jupyter notebook which you can download and run on your machine for a local demo of the Face Frontalization model. The following code downloads (and loads!) the trained PyTorch generative model and the *network* module that are needed at inference time, from the [Hugging Face Hub model repo](https://huggingface.co/opetrova/face-frontalization):

```
from huggingface_hub import hf_hub_url, hf_hub_download

path_to_model = hf_hub_download(repo_id="opetrova/face-frontalization", filename="generator_v0.pt")

# Download network.py into the current directory
network_url = hf_hub_url(repo_id="opetrova/face-frontalization", filename="network.py")
r = requests.get(network_url, allow_redirects=True)
open('network.py', 'wb').write(r.content)

saved_model = torch.load(path_to_model, map_location=torch.device('cpu'))

```

(Naturally, the same code can be used outside Jupyter for whatever use of the trained Face Frontalization model that you see fit.)
