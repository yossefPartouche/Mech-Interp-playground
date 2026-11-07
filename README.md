## Setup

Clone and enter the project:
```bash
cd ~/Projects/mech-interp-playground
```

Create and activate a virtual environment (only needed once):
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Every time you sit down to work
```bash
cd ~/Projects/mech-interp-playground
source venv/bin/activate
code .
```

Check the bottom-right corner of VS Code to confirm the interpreter is `./venv/bin/python`.

## Updating requirements.txt
After installing a new package:
```bash
pip freeze > requirements.txt
```

### Advice from someone who knows nothing but is passionately curious 

I'm writing this in a period where all of this could be done in a single prompt flawlessly. 
For anyone learning this, it's easy to fall prey to being over-assisted by LLMs and lose the essence of learning. 
So try to imagine this journey like a PS2 adventure game (that's what I'm doing now), each section is a mini mission that you'll try to accomplish some easy some hard and when you're tired of playing STOP (so that it becomes a fun and memorable activity to return to rather than a task).
