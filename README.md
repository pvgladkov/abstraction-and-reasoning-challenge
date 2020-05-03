
```bash
$ docker build -t arc -f Dockerfile .
$ docker run -v /var/local/pgladkov/data:/data -v /var/local/pgladkov/abstraction-and-reasoning-challenge:/app --runtime nvidia -it arc 
$ kaggle competitions download -c abstraction-and-reasoning-challenge -p /data/arc/

```

### Run

```bash
$ python arc_run.py
```