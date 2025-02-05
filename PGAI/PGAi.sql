CREATE EXTENSION IF NOT EXISTS ai CASCADE;

SELECT ai.load_dataset('wikimedia/wikipedia', '20231101.en', table_name=>'wiki', batch_size=>5, max_batches=>3);

ALTER TABLE wiki ADD PRIMARY KEY (id);

SELECT * from wiki

SELECT ai.create_vectorizer(
     'wiki'::regclass,
     destination => 'wiki_embeddings',
     embedding => ai.embedding_ollama('nomic-embed-text', 768),
     chunking => ai.chunking_recursive_character_text_splitter('text')
);

select * from ai.vectorizer_status;

SELECT title, chunk
FROM wiki_embeddings 
ORDER BY embedding <=> ai.ollama_embed('nomic-embed-text', 'properties of light')
LIMIT 3;

SELECT title, chunk
FROM wiki_embeddings 
ORDER BY embedding <=> ai.ollama_embed('nomic-embed-text', 'AI tools')
LIMIT 3;CREATE EXTENSION IF NOT EXISTS ai CASCADE;

SELECT ai.load_dataset('wikimedia/wikipedia', '20231101.en', table_name=>'wiki', batch_size=>5, max_batches=>3);

ALTER TABLE wiki ADD PRIMARY KEY (id);

SELECT * from wiki

SELECT ai.create_vectorizer(
     'wiki'::regclass,
     destination => 'wiki_embeddings',
     embedding => ai.embedding_ollama('nomic-embed-text', 768),
     chunking => ai.chunking_recursive_character_text_splitter('text')
);

select * from ai.vectorizer_status;

SELECT title, chunk
FROM wiki_embeddings 
ORDER BY embedding <=> ai.ollama_embed('nomic-embed-text', 'properties of light')
LIMIT 3;

SELECT title, chunk
FROM wiki_embeddings 
ORDER BY embedding <=> ai.ollama_embed('nomic-embed-text', 'AI tools')
LIMIT 3;