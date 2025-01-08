# Semantic Search For LHCb papers 

This project is providing a semantic search for LHCb papers. It provides a pipeline to search by similar papers or use natural language queries to search for papers. It uses embeddings to find similar papers and sentence transformers to encode the papers and queries.

There is a web interface to search for papers in easy and straightforward way. I plan to update the papers database every week to keep the search results up to date. There are many papers published by LHCb collaboration and about LHCb every week on arxiv. 

This is currently more of a proof of concept and I plan to improve the search results and the interface in the future. I'm going to present the results of this projects in a more technical manner later. 

a demo of the project is available on my website hosted on a humble server [here](https://lhcbfinder.net/)

There are two main parts of this project:

## pipeline

The code inside the `model` folder is responsible for creating the embeddings for the papers. It scrapes the arxiv papers dataset on kaggle and uses sentence transformers to encode the papers. The embeddings are saved in a file and used in the search part. I also use `Pinecode` to create the embeddings for the papers.

## website 

The code inside the `website` folder is responsible for the web interface. It uses the embeddings created by the pipeline to search for similar papers or use natural language queries to search for papers. The website is built using `Flask` python library.


## How to run the code

TBD

## Future work

A lot of work on quality and improving the embeddings and the search results. I also plan to add more features to the website and make it more user friendly. if you like to contribute to this project, feel free to open an issue or a pull request.

## Acknowledgements
This work is a fork of [searchthearxiv](https://github.com/augustwester/searchthearxiv) project that provides a semantic search for Machine Leaning arxiv papers. I used the code and the idea to create a similar project for LHCb papers.

## License
This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details.