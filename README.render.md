# LibreChat Deployment Guide for Render.com

## Introduction
This guide explains how to deploy LibreChat on Render.com using the provided render.yaml blueprint file.

## Deployment Steps

1. Fork or clone the LibreChat repository
2. Sign up for a [Render.com](https://render.com) account
3. Create a new Blueprint instance from your repository
4. Deploy

## Required Environment Variables

After deployment, you'll need to set these environment variables in the LibreChat web service:

### Service Hostnames
For each service, you need to find the internal hostname:

1. Go to the Render.com Dashboard
2. Select the service (e.g., vectordb)
3. Click "Connect" in the top right
4. Look for the internal hostname in the "Internal" tab

Set these values in the LibreChat service's environment variables:

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `MONGODB_HOST` | Internal hostname for MongoDB | `mongodb-abc123` |
| `MEILISEARCH_HOST` | Internal hostname for MeiliSearch | `meilisearch-xyz456` |
| `VECTORDB_HOST` | Internal hostname for Vector DB | `vectordb-def789` |
| `RAG_API_HOST` | Internal hostname for RAG API | `rag-api-ghi012` |

### API Keys

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (for embeddings and models) |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (optional) |
| `GROQ_API_KEY` | Your Groq API key (optional) |
| `MISTRAL_API_KEY` | Your Mistral API key (optional) |

## Troubleshooting

- **Connection errors**: Make sure the internal hostnames are correctly set
- **Database initialization**: The first startup may take some time as the databases initialize
- **Authentication issues**: Ensure all API keys are correctly set
- **Memory errors**: Consider upgrading to a higher-tier plan if you encounter memory limits

## Additional Configuration

Once deployed, you'll need to:

1. Create an initial admin user via the web interface
2. Configure any additional LLM endpoints you want to use
3. Upload your initial documents for RAG capabilities

For more details, see the [LibreChat Documentation](https://docs.librechat.ai/).