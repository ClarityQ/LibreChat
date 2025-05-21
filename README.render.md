# LibreChat Deployment Guide for Render.com

## Introduction
This guide explains how to deploy LibreChat on Render.com using the provided render.yaml blueprint file.

## Deployment Steps

1. Fork or clone the LibreChat repository
2. Sign up for a [Render.com](https://render.com) account
3. Create a new Blueprint instance from your repository
4. Deploy

## Required API Keys

After deployment, you'll need to set these API keys in the environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (for embeddings and models) |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (optional) |
| `GROQ_API_KEY` | Your Groq API key (optional) |
| `MISTRAL_API_KEY` | Your Mistral API key (optional) |

## Service Communication

Services communicate with each other using internal domains:

- MongoDB: `mongodb.internal:27017`
- MeiliSearch: `meilisearch.internal:7700`
- Vector Database: `vectordb.internal:5432`
- RAG API: `rag-api.internal:8000`

These domains are automatically set up by the render.yaml blueprint.

## Troubleshooting

- **Connection errors**: Check the service logs for specific error messages
- **Database initialization**: The first startup may take some time as the databases initialize
- **Authentication issues**: Ensure all API keys are correctly set
- **Memory errors**: Consider upgrading to a higher-tier plan if you encounter memory limits

## Additional Configuration

Once deployed, you'll need to:

1. Create an initial admin user via the web interface
2. Configure any additional LLM endpoints you want to use
3. Upload your initial documents for RAG capabilities

For more details, see the [LibreChat Documentation](https://docs.librechat.ai/).