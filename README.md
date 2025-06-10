# Semantic Search For LHCb papers 

This project is providing a semantic search for LHCb papers. It provides a pipeline to search by similar papers or use natural language queries to search for papers. It uses embeddings to find similar papers and sentence transformers to encode the papers and queries.

There is a web interface to search for papers in easy and straightforward way. I plan to update the papers database every week to keep the search results up to date. There are many papers published by LHCb collaboration and about LHCb every week on arxiv. 

This is currently more of a proof of concept and I plan to improve the search results and the interface in the future. I'm going to present the results of this projects in a more technical manner later. 

a demo of the project is available and hosted on a humble server [here](https://lhcbfinder.net/)

There are two main parts of this project:

## model

The code inside the `model` folder is responsible for creating the embeddings for the papers. It scrapes the arxiv papers dataset on kaggle and uses sentence transformers to encode the papers. The embeddings are saved in a file and used in the search part. I also use `Pinecone` to create the embeddings for the papers.

## website 

The code inside the `website` folder is responsible for the web interface. It uses the embeddings created by the pipeline to search for similar papers or use natural language queries to search for papers. The website is built using `Flask` python library.


## How to run the code

### Website Setup

The website component can be run either locally using Python or via Docker. Both methods are explained below.

#### Prerequisites
- Python `3.10` or higher
- Pinecone API key and index name

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/MohamedElashri/lhcbfinder.git
cd lhcbfinder/website
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate # Linux/Mac, for Windows use venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the website directory with your Pinecone credentials:
```env
PINECONE_API_KEY=your_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
FLASK_ENV=development
```

5. Run the development server:
```bash
python run_local.py
```

The website should now be accessible at `http://localhost:8000`

### Docker Deployment

For production deployment, you can use Docker and docker-compose:

1. Clone the repository:
```bash
git clone https://github.com/MohamedElashri/lhcbfinder
cd lhcb-search/website
```

2. Create a `.env` file with your credentials:
```env
PINECONE_API_KEY=your_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
FLASK_ENV=production
FLASK_APP=app.py
```

3. Build and start the containers:
```bash
docker compose up -d
```

The website will be available at `http://localhost:8000`

### Configuration Options

#### Environment Variables
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- `FLASK_ENV`: Set to 'development' for local development or 'production' for deployment
- `WORKERS`: Number of Gunicorn workers (default: 2)
- `THREADS`: Number of threads per worker (default: 4)
- `TIMEOUT`: Worker timeout in seconds (default: 60)

#### Rate Limiting
The application includes rate limiting to prevent abuse:
- 1 request per 30 seconds per IP
- 5 requests per 3 minutes per IP


### Troubleshooting

1. If you see Redis connection errors in Docker:
   - Ensure Redis container is running: `docker-compose ps`
   - Check Redis logs: `docker-compose logs redis`

2. If rate limiting is too restrictive for development:
   - Use development environment to switch to memory-based storage
   - Adjust limits in `app.py` if needed

3. For connection issues with Pinecone:
   - Verify your API key and index name
   - Check your network connection
   - Ensure your Pinecone plan is active

## Future work

A lot of work on quality and improving the embeddings and the search results. I also plan to add more features to the website and make it more user friendly. if you like to contribute to this project, feel free to open an issue or a pull request.

## Acknowledgements
This work is a fork of [searchthearxiv](https://github.com/augustwester/searchthearxiv) project that provides a semantic search for Machine Leaning arxiv papers. I used the code and the idea to create a similar project for LHCb papers.

## License
This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details.
