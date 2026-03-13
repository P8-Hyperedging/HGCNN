Husk at lægge yelp_dataset ind under nn/data mappen ellers virker lortet ikke :D

Run it (surely):

`python -m venv .venv`

`source .venv/bin/activate   # Windows: venv312\Scripts\activate`

`pip install -r nn/requirements.txt`

`cd nn`

`python train.py`

Run all pytests (from HGCNN/nn folder):

`pytest -v -s`