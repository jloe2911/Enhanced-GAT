# SemREC2023

1. Download [OWL2Bench](https://github.com/semrec/semrec.github.io/tree/main/Datasets_SemREC2022/ORE).
2. Download [ORE](https://github.com/semrec/semrec.github.io/tree/main/Datasets_SemREC2022/OWL2Bench).
3. Download [CaLiGraph](https://data.dws.informatik.uni-mannheim.de/CaLiGraph/CaLiGraph-for-SemREC/SemREC-2022-Datasets/).
4. Run ```train.ipynb``` to train the following models:

* Graph Attention Network (GAT).
* 2-Hop Graph Attention Network (2-Hop GAT): an extended version of Graph Attention Network that aggregates node information that are 2 hop away. 
* Filtered 2-Hop Graph Attention Network (Filtered 2-Hop GAT): Filtered 2-Hop GAT with a filtering mechanism to include two RDF rules: (1) subclass transitivity: if A is a subclass of B, and B is a subclass of C, then A is also a subclass of C, and (2) if A is a type of B and B is a subclass of C, then A is also a type of C.

5. Run ```eval.ipynb``` to evaluate the models. The metrics are Hits@1 and Hits@10. The trained models can also be found [here](https://drive.google.com/drive/folders/1-PYc8YT9iukTdUkhf-PRbksAxv0jfjq4?usp=sharing).

# Quick Tour

- ```caligraph_full.ipynb```: enables to run the models the entire caligraph ontology. 
- ```axioms.ipynb```: enables to output the number of different axioms for each dataset. 
- ```stats.ipynb```: enables to output the number of different edges and the number of nodes for each dataset. 
