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

### Hardcoded Credentials and Keys

The following values are hardcoded for simplicity:

1. MongoDB:
   - Username: `librechatuser`
   - Password: `librechatpassword`

2. MeiliSearch:
   - Master Key: `librechat-meilisearch-master-key`

3. JWT Authentication:
   - Secret: `librechat-jwt-secret-key-for-render-deployment`

4. Credential Encryption:
   - CREDS_KEY: `librechat-credentials-encryption-key`
   - CREDS_IV: `librechat-creds-iv`

**Note**: In a production environment, you should consider changing these hardcoded values to stronger secrets.

### Required API Keys

These keys must be manually set after deployment:

| Service | Variable | Description |
|---------|----------|-------------|
| LibreChat | `OPENAI_API_KEY` | Your OpenAI API key (for embeddings and models) |
| LibreChat | `ANTHROPIC_API_KEY` | Your Anthropic API key (optional) |
| RAG API | `OPENAI_API_KEY` | Same OpenAI API key (for embeddings in RAG) |

### Optional API Keys

These can be added if you want to use these models:

| Service | Variable | Description |
|---------|----------|-------------|
| LibreChat | `GROQ_API_KEY` | Your Groq API key |
| LibreChat | `MISTRAL_API_KEY` | Your Mistral API key |
| LibreChat | `OPENROUTER_KEY` | Your OpenRouter key |
| LibreChat | `PORTKEY_API_KEY` | Your Portkey API key |

## Troubleshooting

- **Connection errors**: Make sure the internal hostnames are correctly set in environment variables
- **Database initialization**: The first startup may take some time as the databases initialize
- **MongoDB authentication errors**: If you see errors like "Command requires authentication", ensure `MONGODB_PASSWORD` is set correctly and matches in both the MongoDB service and the LibreChat app
- **MeiliSearch authentication errors**: If you see errors like "The provided API key is invalid", ensure `MEILI_KEY` is set and matches in both the MeiliSearch service and the LibreChat app
- **JWT authentication errors**: If there are issues between LibreChat and the RAG API, ensure `JWT_SECRET` is set to the same value for both services
- **Resource limits**: By default, services use the free tier. For production use with higher traffic or memory requirements, you may need to upgrade specific services to paid plans

## Reducing Costs

The blueprint is configured to use Render's free tier where possible:

- No plan specification means services will use the free tier by default
- Free tier has resource limitations (memory, CPU) that might be insufficient for production workloads
- Disk storage still incurs charges, even on free tiers (currently ~$0.10/GB/month)
- For a fully functional development environment, the free tier should work for testing

## Additional Configuration

Once deployed, you'll need to:

1. Create an initial admin user via the web interface
2. Configure any additional LLM endpoints you want to use
3. Upload your initial documents for RAG capabilities

For more details, see the [LibreChat Documentation](https://docs.librechat.ai/).