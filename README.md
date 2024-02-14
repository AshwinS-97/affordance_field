
# Affordance field
This repo builds on top of LeRF( Language embedded Radiance Field ) by adding Affordance field on top of it. 

## File Structure

```
├── affordance
|   ├── data
|   |   ├── utils
|   |   ├── affordance_datamanager.py 
│   ├── __init__.py
│   ├── affordance_config.py
│   ├── affordance_pipeline.py
│   ├── affordance_model.py
│   ├── affordance_field.py
├── pyproject.toml
```

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands: (or use download the venv from [here](--link))

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```

## Running the affordance_model
This repository creates a new Nerfstudio method named "lerf_affordance". To train with it, run the command:
```
ns-train lerf_affordance --data [PATH]
```
