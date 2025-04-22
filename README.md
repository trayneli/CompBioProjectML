
### Workflow
* https://docs.google.com/document/d/1NPdxfBH7t91a6blf6zrmwZCzc-hwK66q9_cPZXo0UZA/edit?usp=sharing

### Project
* Uniprot (annotations, EC number)
* Alphafold --> ProteinMPNN to generate embeddings

### Data Type Processing
- Sequence -> ProtT5 Finetuning -> output EC numbers
- AlphaFold Structure -> ProteinMPNN -> NN(such as transformer or CNN) -> output EC Number
- Functional Annotation -> BioBert -> NN(such as transformer or CNN) -> output EC Number
- Output EC Numbers/Final Weights -> MultiHeadAttention NN -> final output EC Number
