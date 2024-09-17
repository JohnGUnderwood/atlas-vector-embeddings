# How to use
- You must have python3 and pip installed
- Make a copy of `example.env` and rename to `.env`
- Set your environment variables inside the new `.env` file
- Set up a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
source .env
```
- Start changestream
```
python3 changestreams.py
```

Now you can execute the notebook.