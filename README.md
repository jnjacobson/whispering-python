# whispering-python
Python script to transcribe speech to text using OpenAI's Whisper.

## Setup

### Clone the repository

```bash
git clone https://github.com/jnjacobson/whispering-python.git
```

### Add your OpenAI API key

Clone the `.env.dist` file to `.env` and add your OpenAI API key.

```bash
cp .env.dist .env
```

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install system dependencies
```bash
sudo apt install libportaudio2 ffmpeg python3-tk python3-dev
```

## Usage

```bash
uv run script.py
```

### Pass language code as argument

The default language is German.

```bash
uv run script.py --l en
```
