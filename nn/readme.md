Husk at lægge yelp_dataset ind under nn mappen ellers virker lortet ikke :D


Til Viktors sjove eventyr (fantastisk (simpel) hypergraph; bruger som edge, business som node):

"*Hyperedge size = number of businesses reviewed by a user; min: some users reviewed only one business, max: one user reviewed 25 businesses, average: users reviewed ~1.16 businesses in this sample*.

*Hyperedges with size >= 2 create meaningful higher-order structure: Business A -> User -> Business B*"

```python3.9 -m venv venv```

```source venv/bin/activate```

```pip install -r requirements.txt```

```python -m nn.train```
