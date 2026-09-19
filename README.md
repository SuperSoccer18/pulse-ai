# Pulse.AI Proof of Concept

Pulse.AI is a voice-enabled application for creating structured post-call documentation. This proof of concept captures microphone audio, transcribes it using a streaming speech-to-text model, and processes the transcript into a format suitable for downstream review and submission.

## Key Features

* Real-time microphone capture and streaming transcription
* Local speech-to-text inference using Moonshine
* Incremental transcript processing for low latency
* Simple web interface for reviewing generated documentation
* Privacy-conscious design that processes audio in real time without retaining raw recordings

## Technology Stack

* Python
* PyTorch
* Flask
* Moonshine Streaming
* REST APIs

## Model Setup

Download the `moonshine-streaming-medium` model files from [Hugging Face](https://huggingface.co/UsefulSensors/moonshine-streaming-medium/tree/main).

Place the downloaded files in the model directory expected by the application.

Model weights are not included in this repository because of their size.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Code Sample Notes

I developed this proof of concept during my internship at Eli Lilly. My work included evaluating speech-to-text models, implementing the streaming transcription workflow, building the application interface, and integrating the components into an end-to-end prototype.

This repository is a limited code sample. Proprietary integrations, internal configuration, credentials, and company data have been excluded. Any included sample data is synthetic.
