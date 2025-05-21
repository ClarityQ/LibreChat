# LibreChat Deployment Guide for Render.com

## Introduction
This guide explains how to deploy LibreChat on Render.com using the provided render.yaml blueprint file.

## Deployment Steps

1. Fork or clone the LibreChat repository
2. Sign up for a [Render.com](https://render.com) account
3. Create a new Blueprint instance from your repository
4. Fill in the required environment variables (see below)
5. Deploy

## Required Environment Variables

After the initial blueprint deployment, you'll need to manually set these environment variables:

| Variable | Description |
|----------|-------------|
| `VECTORDB_HOST` | Internal hostname of the vectordb service (find in Render Dashboard under "Connect" > "Internal") |
| `MONGODB_HOST` | Internal hostname of the MongoDB service |
| `MEILISEARCH_HOST` | Internal hostname of the Meilisearch service |
| `RAG_API_HOST` | Internal hostname of the RAG API service |
| `OPENAI_API_KEY` | Your OpenAI API key (for embeddings and models) |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (optional) |

## Finding Internal Hostnames

For each service, you need to find the internal hostname:

1. Go to the Render.com Dashboard
2. Select the service (e.g., vectordb)
3. Click "Connect" in the top right
4. Look for the internal hostname in the "Internal" tab (e.g., "vectordb-2j3e")
5. Use this value for the corresponding HOST environment variable

## Troubleshooting

- **Connection errors**: Make sure the internal hostnames are correctly set
- **Startup failures**: Check service logs for specific error messages
- **Database connection issues**: Verify that the PostgreSQL with pgvector service is running before the RAG API service

## Additional Configuration

Once deployed, you'll need to:

1. Create an initial admin user via the web interface
2. Configure any additional LLM endpoints you want to use
3. Upload your initial documents for RAG capabilities

For more details, see the [LibreChat Documentation](https://docs.librechat.ai/).