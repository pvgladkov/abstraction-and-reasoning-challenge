# Abstraction and Reasoning Challenge

#### Create an AI capable of solving reasoning tasks it has never seen before

https://www.kaggle.com/c/abstraction-and-reasoning-challenge


### Build & Run

```bash
$ docker build -t arc -f Dockerfile .
$ docker run -v /var/local/pgladkov/data:/data -v /var/local/pgladkov/abstraction-and-reasoning-challenge:/app --runtime nvidia -it arc 
```

### Download data

```bash
$ kaggle competitions download -c abstraction-and-reasoning-challenge -p /data/arc/
```

### Run

```bash
$ python arc_run.py
```

## Related sources

1. [The Abstraction and Reasoning Corpus](https://github.com/fchollet/ARC).

2. [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547).