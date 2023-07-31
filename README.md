# Debiaser

## Initialization

Build docker:

`docker build -t ai_fellowship_debiaser:latest .
`

Serve the API:

`docker run -v $(pwd)/models/model:/app/model -p 8080:8080 ai_fellowship_debiaser:latest`