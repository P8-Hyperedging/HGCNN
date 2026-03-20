Husk at lægge yelp_dataset ind under nn/data mappen ellers virker lortet ikke :D

Run it (surely):

`python -m venv .venv`

`source .venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1`

`pip install -r ../requirements.txt`

`cd nn`

`python train.py`

Run all pytests (from HGCNN/nn folder):

`pytest -v -s`


# Docker

`docker build -t hgcnn .`

`docker run -p 5000:5000 hgcnn`